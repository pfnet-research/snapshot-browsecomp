"""Evaluates an LLM on the snapshot_browsecomp benchmark.

The model answers each question in a tool-calling loop with two tools over the search corpora: ``search(query,
lang)`` returns the top documents of an OpenSearch query on the corpus of ``lang`` (``en`` or ``ja``), and
``visit(id, offset, length)`` reads a returned document a window at a time. The model picks the corpus of each search
itself, so both indexes are needed whichever language is evaluated. A judge model then compares the final answer
with the gold answer.

The questions are in the format of BrowseComp: one row per question with ``problem``, ``answer`` and optionally
``problem_topic``. When a row has a ``canary``, ``problem`` and ``answer`` are encrypted with it as BrowseComp's
are, and are decrypted here. Each language is one subset (only ``ja`` so far); ``--langs`` picks which are evaluated.

Searches are plain BM25 over ``title`` and ``text``, with no category or domain weighting. The index is the one
``build_index.py`` builds; ``--aoss-host`` searches an Amazon OpenSearch Serverless collection instead.

Results go to ``<output-dir>/<lang>.jsonl``, one line per question, and the accuracies to
``<output-dir>/summary.json``. A question already in ``<lang>.jsonl`` without an error is not asked again, so an
interrupted run is resumed by running the same command.

This file only depends on openai, opensearch-py and datasets (and boto3 for ``--aoss-host``), so that it can be
distributed with the benchmark.

    python evaluate.py --model <model> --base-url <url> --api-key-env <ENV> \\
        --judge-base-url <url> --judge-api-key-env <ENV> --output-dir ./results
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import string
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

QUESTIONS_DATASET = "pfnet/snapshot-browsecomp"
DEFAULT_JUDGE_MODEL = "google/gemma-4-31b-it"
DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
DEFAULT_INDEX = "{lang}-web-search-corpus"
# The languages of the search corpora, and those of the questions.
LANGS = ("en", "ja")
QUESTION_LANGS = ("ja",)
DIFFICULTIES = ("lite", "hard")

DEFAULT_MAX_TOOL_CALLS = 40
DEFAULT_MAX_RESULTS = 10
MAX_VISIT_CHARS = 8 * 1024
SNIPPET_CHARS = 200
# Rounds the loop may run beyond `max_tool_calls`, for a model that keeps calling tools after the budget is spent.
_ROUND_SAFETY_MARGIN = 2
_SEARCH_FIELDS = ["title", "text"]

ANSWER_TAG = "answer"
VERDICT_TAG = "verdict"

SOLVE_PROMPT = string.Template(
    """You are answering a challenging trivia question that may take several web searches to solve.

Question: ${question}

Use the `search` and `visit` tools to find the information required to answer. You may make at most \
${max_tool_calls} tool calls in total, counting `search` and `visit` together; plan for that budget and \
commit to an answer before you run out, because once it is spent you will be asked for your answer with \
the tools withdrawn. Do not guess before you have searched -- the question is designed to require \
looking things up.

Base your answer on what the pages `search` returns say, even where they disagree with what you know, \
and read "now", "currently" or "the latest" in the question as of the time of those pages.

`search` takes a required `lang` argument: `en` searches English pages and `ja` Japanese pages. Write \
the query text itself in that language.

`visit` takes the `id` of one of `search`'s results -- the `id` that search call returned for it, not its \
URL. Only an id a previous `search` returned can be visited -- there is no way to fetch an arbitrary URL. \
`visit(id, offset, length)` returns up to ${max_visit_chars} characters of a page at a time, starting at \
character `offset` (0, the beginning, if omitted). A page is often longer than that: the result gives the \
page's total length as `n_chars` and, when there is more after what was returned, the `next_offset` to call \
`visit` with to read on. Each such call is one tool call.

When you are ready to commit to a final answer, respond with your reasoning, then end your response \
with a <${tag}> block containing a JSON object in this exact schema:

<${tag}>
{"answer": "<your final answer, as short and specific as possible>"}
</${tag}>

Do not call any more tools once you give your final answer. Do not write anything after the closing \
</${tag}> tag."""
)

OUT_OF_BUDGET_PROMPT = string.Template(
    """You have used all of your tool calls. Using what you have found so far, give your final answer now, \
ending your response with the <${tag}> block described above."""
)

# The prompt of `--no-tools`, which asks for the answer from the model's own knowledge.
NO_TOOLS_PROMPT = string.Template(
    """You are answering a challenging trivia question.

Question: ${question}

Respond with your reasoning, then end your response with a <${tag}> block containing a JSON object in this \
exact schema:

<${tag}>
{"answer": "<your final answer, as short and specific as possible>"}
</${tag}>

Do not write anything after the closing </${tag}> tag."""
)

# The LLM-as-a-judge prompt of EvoBrowseComp (arXiv 2606.13120v1, Appendix D), with its three sections asked for
# as one JSON object.
JUDGE_PROMPT = string.Template(
    """You are a general artificial intelligence assistant. Based on the [Correct Answer] provided \
below, judge whether the [Response] to the [Original Question] below is correct.

[Original Question]: ${question}

[Correct Answer]: ${gold_answer}

[Response]: ${candidate_answer}

Your judgment must follow the following format and standards:

final_answer: The final accurate answer extracted from the [Response]. If there is no clear final \
answer in the [Response], fill in 'None'.
explanation: Explain why the [final_answer] is correct or incorrect based on the [Correct Answer]. Only \
focus on whether there is a substantial difference between the [final_answer] and the [Correct Answer]; \
do not comment on the background of the question, do not attempt to re-solve the problem, do not defend \
any answer different from the [Correct Answer], and only focus on judging whether the answers are \
consistent.
conclusion: If the [final_answer] is consistent with the [Correct Answer] provided above, or within an \
acceptable small error range for numerical questions, fill in 'Correct'; otherwise (i.e., in case of \
any inconsistency, ambiguity, inequivalence, or incorrect extracted answer), fill in 'Incorrect'.

Respond with your reasoning, then give your final judgment as a <${tag}> block containing the JSON \
object and nothing else:

<${tag}>
{"final_answer": "...", "explanation": "...", "conclusion": "Correct | Incorrect"}
</${tag}>

Do not write anything after the closing </${tag}> tag."""
)

# --- Questions ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    id: str
    lang: str
    problem: str
    answer: str
    topic: str = ""
    difficulty: str = ""


def decrypt(ciphertext_b64: str, password: str) -> str:
    """BrowseComp's decryption: XOR with the SHA-256 of ``password``, repeated to the length of the text."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = hashlib.sha256(password.encode()).digest()
    key = (key * (len(encrypted) // len(key) + 1))[: len(encrypted)]
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def to_question(row: dict[str, Any], lang: str, index: int) -> Question:
    problem, answer = row["problem"], row["answer"]
    if row.get("canary"):
        problem, answer = decrypt(problem, row["canary"]), decrypt(answer, row["canary"])
    return Question(
        id=str(row.get("id") or index),
        lang=lang,
        problem=problem,
        answer=answer,
        topic=row.get("problem_topic") or "",
        difficulty=row.get("difficulty") or "",
    )


def load_questions(source: str, lang: str, split: str, difficulty: str = "both") -> list[Question]:
    """The questions of ``lang``: ``source`` is a HF dataset id, or a local file whose path may contain ``{lang}``.

    ``difficulty`` other than ``both`` keeps only the questions of that difficulty.
    """
    import datasets

    path = source.format(lang=lang)
    if os.path.exists(path):
        builder = "parquet" if path.endswith(".parquet") else "csv" if path.endswith(".csv") else "json"
        ds = datasets.load_dataset(builder, data_files=path, split="train")
    else:
        ds = datasets.load_dataset(source, lang, split=split)
    questions = [to_question(row, lang, i) for i, row in enumerate(ds)]
    return [q for q in questions if difficulty == "both" or q.difficulty == difficulty]


# --- Search ------------------------------------------------------------------------------------------------------


def _phrase_condition(word: str, lang: str) -> dict[str, Any]:
    if lang == "ja":
        return {
            "dis_max": {
                "queries": [
                    {
                        "multi_match": {
                            "query": word,
                            "fields": [f"{f}.analyzed^5" for f in _SEARCH_FIELDS],
                            "type": "phrase",
                            "analyzer": "ja_kuromoji_analyzer",
                        }
                    },
                    {
                        "multi_match": {
                            "query": word,
                            "fields": [f"{f}.ngram^1" for f in _SEARCH_FIELDS],
                            "type": "phrase",
                            "analyzer": "ja_ngram_index_analyzer",
                        }
                    },
                ]
            }
        }
    return {"multi_match": {"query": word, "type": "phrase", "fields": _SEARCH_FIELDS}}


def search_body(query: str, lang: str, size: int) -> dict[str, Any]:
    """Each whitespace-separated word of ``query`` is a phrase; documents with all of them rank above the rest."""
    words = query.split()
    if not words:
        raise ValueError("the query is empty")
    conditions = [_phrase_condition(word, lang) for word in words]
    suffixes = [".analyzed", ".ngram"] if lang == "ja" else [""]
    return {
        "query": {"bool": {"should": [{"bool": {"must": conditions}}, {"bool": {"should": conditions}}]}},
        "size": size,
        "_source": ["title", "text", "url"],
        "highlight": {
            "fields": {f"text{suffix}": {} for suffix in suffixes},
            "fragment_size": SNIPPET_CHARS,
            "number_of_fragments": 1,
        },
    }


@dataclass(frozen=True)
class Document:
    title: str
    url: str
    text: str
    snippet: str


def _snippet(hit: dict[str, Any], text: str) -> str:
    for fragments in (hit.get("highlight") or {}).values():
        if fragments:
            return str(fragments[0])
    return text[:SNIPPET_CHARS]


class Searcher:
    """Searches the corpus of each language with a synchronous opensearch-py client, in threads."""

    def __init__(self, client: Any, index: str, max_results: int) -> None:
        self._client = client
        self._index = index
        self._max_results = max_results

    def _search(self, query: str, lang: str) -> list[Document]:
        response = self._client.search(
            index=self._index.format(lang=lang), body=search_body(query, lang, self._max_results)
        )
        docs = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            text = source.get("text") or ""
            docs.append(
                Document(
                    title=source.get("title") or "", url=source.get("url") or "", text=text, snippet=_snippet(hit, text)
                )
            )
        return docs

    async def search(self, query: str, lang: str) -> list[Document]:
        return await asyncio.to_thread(self._search, query, lang)


def make_opensearch_client(url: str, aoss_host: str | None, aws_region: str) -> Any:
    from opensearchpy import OpenSearch, RequestsHttpConnection

    if aoss_host is None:
        return OpenSearch(hosts=[url], http_compress=True, timeout=120, retry_on_timeout=True, max_retries=3)
    import boto3
    from opensearchpy import AWSV4SignerAuth

    credentials = boto3.Session().get_credentials()
    return OpenSearch(
        hosts=[{"host": aoss_host, "port": 443}],
        http_auth=AWSV4SignerAuth(credentials, aws_region, "aoss"),
        use_ssl=True,
        verify_certs=True,
        http_compress=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=32,
        timeout=120,
        retry_on_timeout=True,
        max_retries=3,
    )


# --- Agent -------------------------------------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web and return a list of results, each with an id, title, snippet, and URL. "
                "Use `visit` with a result's id to read its page content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query to issue."},
                    "lang": {
                        "type": "string",
                        "description": (
                            "Language code for the search. Use `en` for English or `ja` for Japanese; "
                            "unknown codes fall back to `en`."
                        ),
                    },
                },
                "required": ["query", "lang"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": (
                f"Visit a search result and return its page content, {MAX_VISIT_CHARS} characters at a time. "
                "The result has the page's total length (`n_chars`) and, when the page goes on, the `next_offset` "
                "to pass as `offset` to read the next part."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The id of a result a previous `search` returned."},
                    "offset": {
                        "type": "integer",
                        "description": "Character position in the page to start reading from. Default 0.",
                    },
                    "length": {
                        "type": "integer",
                        "description": f"Number of characters to return, at most {MAX_VISIT_CHARS} (the default).",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    },
]


def _int_argument(arguments: dict[str, Any], name: str, default: int) -> int:
    value = arguments.get(name)
    if value is None:
        return default
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"`{name}` must be an integer")
    return value


def page_window(text: str, arguments: dict[str, Any]) -> dict[str, Any]:
    offset = _int_argument(arguments, "offset", 0)
    length = _int_argument(arguments, "length", MAX_VISIT_CHARS)
    if offset < 0:
        raise ValueError("`offset` must not be negative")
    if length <= 0:
        raise ValueError("`length` must be positive")
    if offset > 0 and offset >= len(text):
        raise ValueError(f"`offset` {offset} is past the end of the page, which has {len(text)} characters")
    end = min(len(text), offset + min(length, MAX_VISIT_CHARS))
    window: dict[str, Any] = {"content": text[offset:end], "offset": offset, "n_chars": len(text)}
    if end < len(text):
        window["next_offset"] = end
    return window


class ToolSession:
    """The tools of one question: the documents its searches returned, by the ids given to the model."""

    def __init__(self, searcher: Searcher) -> None:
        self._searcher = searcher
        self._docs: dict[str, Document] = {}

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("`query` must be a non-empty string")
            lang = arguments.get("lang")
            results = []
            for doc in await self._searcher.search(query, lang if lang in LANGS else "en"):
                id = f"d{len(self._docs) + 1}"
                self._docs[id] = doc
                results.append({"id": id, "title": doc.title, "url": doc.url, "snippet": doc.snippet})
            return {"results": results}
        if name == "visit":
            id = str(arguments.get("id", "")).strip()
            if id not in self._docs:
                raise ValueError(f"`{id}` is not an id a `search` has returned")
            doc = self._docs[id]
            return {"url": doc.url, "title": doc.title, **page_window(doc.text, arguments)}
        raise ValueError(f"unknown tool `{name}`")


@dataclass(frozen=True)
class ModelConfig:
    model: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def sampling(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model}
        for name in ("max_tokens", "temperature", "top_p"):
            if getattr(self, name) is not None:
                kwargs[name] = getattr(self, name)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs


@dataclass
class AgentResult:
    final_message: str
    tool_call_count: int
    exceeded_tool_limit: bool
    messages: list[dict[str, Any]]


# Attempts of a completion whose response has no choices: OpenRouter returns its errors that way, with status 200,
# so the client does not retry them itself.
_COMPLETION_ATTEMPTS = 5


async def complete(client: Any, **kwargs: Any) -> Any:
    """``client.chat.completions.create``, asked again with a backoff while the response has no choices."""
    for attempt in range(_COMPLETION_ATTEMPTS):
        response = await client.chat.completions.create(**kwargs)
        if response.choices:
            return response
        if attempt + 1 < _COMPLETION_ATTEMPTS:
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"no choices in {_COMPLETION_ATTEMPTS} attempts: {getattr(response, 'error', None)}")


def _valid_json_arguments(arguments: str | None) -> str:
    """``arguments`` as is when they are valid JSON, else ``{}``: some providers reject a history with invalid ones."""
    try:
        json.loads(arguments or "")
    except json.JSONDecodeError:
        return "{}"
    return arguments or "{}"


def _assistant_message(message: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.function.name, "arguments": _valid_json_arguments(c.function.arguments)},
            }
            for c in message.tool_calls
        ]
    return out


async def run_agent(
    client: Any, config: ModelConfig, question: Question, tools: ToolSession, *, max_tool_calls: int
) -> AgentResult:
    prompt = SOLVE_PROMPT.substitute(
        question=question.problem,
        max_tool_calls=max_tool_calls,
        max_visit_chars=f"{MAX_VISIT_CHARS:,}",
        tag=ANSWER_TAG,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_call_count = 0
    for _ in range(max_tool_calls + _ROUND_SAFETY_MARGIN):
        response = await complete(client, messages=messages, tools=TOOLS, tool_choice="auto", **config.sampling())
        message = response.choices[0].message
        messages.append(_assistant_message(message))
        if not message.tool_calls:
            return AgentResult(message.content or "", tool_call_count, False, messages)
        exceeded = False
        for call in message.tool_calls:
            if tool_call_count >= max_tool_calls:
                exceeded = True
                result: dict[str, Any] = {"error": "tool call limit exceeded"}
            else:
                tool_call_count += 1
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("the arguments must be a JSON object")
                    result = await tools.call(call.function.name, arguments)
                except Exception as e:  # noqa: BLE001 -- a failed call is the model's to read
                    result = {"error": f"{type(e).__name__}: {e}"}
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)}
            )
        if exceeded:
            break
    # Out of budget: ask for the answer with the tools withdrawn.
    messages.append({"role": "user", "content": OUT_OF_BUDGET_PROMPT.substitute(tag=ANSWER_TAG)})
    response = await complete(client, messages=messages, **config.sampling())
    message = response.choices[0].message
    messages.append(_assistant_message(message))
    return AgentResult(message.content or "", tool_call_count, True, messages)


async def run_without_tools(client: Any, config: ModelConfig, question: Question) -> AgentResult:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": NO_TOOLS_PROMPT.substitute(question=question.problem, tag=ANSWER_TAG)}
    ]
    response = await complete(client, messages=messages, **config.sampling())
    message = response.choices[0].message
    messages.append(_assistant_message(message))
    return AgentResult(message.content or "", 0, False, messages)


# The most of a response the judge is given as its answer, from the end, when it has no usable ``<answer>`` block.
MAX_UNTAGGED_ANSWER_CHARS = 2000


def extract_final_answer(text: str) -> str:
    """The answer in the ``<answer>`` block, or the end of the text when there is none, so that it can still be judged."""
    block = text.partition(f"<{ANSWER_TAG}>")[2].partition(f"</{ANSWER_TAG}>")[0].strip()
    if not block:
        return text.strip()[-MAX_UNTAGGED_ANSWER_CHARS:]
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        return block[-MAX_UNTAGGED_ANSWER_CHARS:]
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str) and parsed["answer"].strip():
        return parsed["answer"].strip()
    return block


# --- Judge -------------------------------------------------------------------------------------------------------


def parse_verdict(text: str) -> tuple[bool, str]:
    block = text.partition(f"<{VERDICT_TAG}>")[2].partition(f"</{VERDICT_TAG}>")[0]
    if not block.strip():
        raise ValueError(f"the response has no <{VERDICT_TAG}> block")
    parsed = json.loads(block)
    if not isinstance(parsed, dict):
        raise ValueError(f"<{VERDICT_TAG}> must be a JSON object")
    conclusion = parsed.get("conclusion")
    if not isinstance(conclusion, str) or conclusion.strip().lower() not in ("correct", "incorrect"):
        raise ValueError('`conclusion` must be "Correct" or "Incorrect"')
    return conclusion.strip().lower() == "correct", str(parsed.get("explanation") or "")


async def judge(
    client: Any, config: ModelConfig, question: Question, answer: str, *, n_attempts: int = 3
) -> tuple[bool, str]:
    """``(correct, explanation)``; raises when no attempt produced a verdict."""
    prompt = JUDGE_PROMPT.substitute(
        question=question.problem, gold_answer=question.answer, candidate_answer=answer, tag=VERDICT_TAG
    )
    error: Exception | None = None
    for _ in range(n_attempts):
        response = await complete(client, messages=[{"role": "user", "content": prompt}], **config.sampling())
        try:
            return parse_verdict(response.choices[0].message.content or "")
        except (ValueError, json.JSONDecodeError) as e:
            error = e
    raise ValueError(f"the judge gave no verdict in {n_attempts} attempts: {error}")


# --- Evaluation --------------------------------------------------------------------------------------------------


@dataclass
class Record:
    id: str
    lang: str
    problem: str
    gold_answer: str
    topic: str
    difficulty: str = ""
    predicted_answer: str = ""
    correct: bool = False
    judge_explanation: str = ""
    tool_call_count: int = 0
    exceeded_tool_limit: bool = False
    elapsed_seconds: float = 0.0
    # Non-empty when the question could not be answered or judged; such a question is asked again on resume.
    error: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)


async def evaluate_question(
    question: Question,
    *,
    model_client: Any,
    model_config: ModelConfig,
    judge_client: Any,
    judge_config: ModelConfig,
    searcher: Searcher | None,
    max_tool_calls: int,
) -> Record:
    """Answers ``question`` with the search tools, or without them when ``searcher`` is None, and judges it."""
    record = Record(
        id=question.id,
        lang=question.lang,
        problem=question.problem,
        gold_answer=question.answer,
        topic=question.topic,
        difficulty=question.difficulty,
    )
    start = time.monotonic()
    try:
        if searcher is None:
            result = await run_without_tools(model_client, model_config, question)
        else:
            result = await run_agent(
                model_client, model_config, question, ToolSession(searcher), max_tool_calls=max_tool_calls
            )
        record.messages = result.messages
        record.tool_call_count = result.tool_call_count
        record.exceeded_tool_limit = result.exceeded_tool_limit
        record.predicted_answer = extract_final_answer(result.final_message)
        record.correct, record.judge_explanation = await judge(
            judge_client, judge_config, question, record.predicted_answer
        )
    except Exception as e:  # noqa: BLE001 -- recorded, and asked again on resume
        record.error = f"{type(e).__name__}: {e}"
    record.elapsed_seconds = time.monotonic() - start
    return record


def read_records(path: str) -> dict[str, dict[str, Any]]:
    """The last record of each question in ``path``."""
    records: dict[str, dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records[record["id"]] = record
    return records


def summarize(records_by_lang: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(records)
        n_correct = sum(1 for r in records if r["correct"] and not r["error"])
        return {
            "n": n,
            "n_correct": n_correct,
            "accuracy": n_correct / n if n else 0.0,
            "n_error": sum(1 for r in records if r["error"]),
            "mean_tool_calls": sum(r["tool_call_count"] for r in records) / n if n else 0.0,
        }

    summary = {lang: stats(records) for lang, records in records_by_lang.items()}
    if len(records_by_lang) > 1:
        summary["all"] = stats([r for records in records_by_lang.values() for r in records])
    return summary


async def evaluate(
    questions: list[Question],
    output_path: str,
    run_one: Callable[[Question], Awaitable[Record]],
    *,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Answers the questions not yet in ``output_path`` and appends their records; returns every question's record."""
    done = {id for id, r in read_records(output_path).items() if not r["error"]}
    todo = [q for q in questions if q.id not in done]
    print(f"{output_path}: {len(questions) - len(todo)} done, {len(todo)} to go", flush=True)
    semaphore = asyncio.Semaphore(concurrency)
    finished = 0

    async def run(question: Question) -> None:
        nonlocal finished
        async with semaphore:
            record = asdict(await run_one(question))
        # No await between opening and closing the file, so concurrent writers do not interleave.
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        finished += 1
        status = record["error"] or ("correct" if record["correct"] else "incorrect")
        print(f"[{finished}/{len(todo)}] {question.lang} {question.id}: {status}", flush=True)

    await asyncio.gather(*(run(q) for q in todo))
    records = read_records(output_path)
    return [records[q.id] for q in questions]


def _api_key(env: str | None, path: str | None) -> str:
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    if env:
        return os.environ[env]
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


def _openai_client(base_url: str | None, api_key: str, timeout: float) -> Any:
    from openai import AsyncOpenAI

    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--questions", type=str, default=QUESTIONS_DATASET, help="HF dataset id, or a local file ({lang} is replaced)"
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--langs", type=str, nargs="+", default=list(QUESTION_LANGS), choices=QUESTION_LANGS)
    parser.add_argument("--difficulty", type=str, default="both", choices=DIFFICULTIES + ("both",))
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions of each language")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.0, help="Seconds an LLM request may take")
    # The model evaluated.
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None, help="Environment variable holding the API key")
    parser.add_argument("--api-key-file", type=str, default=None, help="File holding the API key")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--extra-body", type=json.loads, default={}, help="JSON object passed to the API as is")
    parser.add_argument("--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS)
    parser.add_argument("--no-tools", action="store_true", help="Answer without the search tools")
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS, help="Results per search")
    # The judge.
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--judge-api-key-env", type=str, default=None)
    parser.add_argument("--judge-api-key-file", type=str, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    # The search engine.
    parser.add_argument("--opensearch-url", type=str, default=DEFAULT_OPENSEARCH_URL)
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX, help="Index name; {lang} is replaced")
    parser.add_argument("--aoss-host", type=str, default=None, help="Search this OpenSearch Serverless host instead")
    parser.add_argument("--aws-region", type=str, default="ap-northeast-1")
    args = parser.parse_args()

    model_client = _openai_client(args.base_url, _api_key(args.api_key_env, args.api_key_file), args.timeout)
    judge_client = _openai_client(
        args.judge_base_url, _api_key(args.judge_api_key_env, args.judge_api_key_file), args.timeout
    )
    model_config = ModelConfig(args.model, args.max_tokens, args.temperature, args.top_p, args.extra_body)
    judge_config = ModelConfig(args.judge_model, args.judge_max_tokens, temperature=0.0)
    searcher = (
        None
        if args.no_tools
        else Searcher(
            make_opensearch_client(args.opensearch_url, args.aoss_host, args.aws_region), args.index, args.max_results
        )
    )

    async def run_one(question: Question) -> Record:
        return await evaluate_question(
            question,
            model_client=model_client,
            model_config=model_config,
            judge_client=judge_client,
            judge_config=judge_config,
            searcher=searcher,
            max_tool_calls=args.max_tool_calls,
        )

    async def run_all() -> dict[str, list[dict[str, Any]]]:
        records_by_lang = {}
        for lang in args.langs:
            questions = load_questions(args.questions, lang, args.split, args.difficulty)[: args.limit]
            output_path = os.path.join(args.output_dir, f"{lang}.jsonl")
            records_by_lang[lang] = await evaluate(questions, output_path, run_one, concurrency=args.concurrency)
        return records_by_lang

    os.makedirs(args.output_dir, exist_ok=True)
    summary = summarize(asyncio.run(run_all()))
    summary["config"] = {k: v for k, v in vars(args).items() if "api_key" not in k}
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
