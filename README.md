# SnapshotBrowseComp

SnapshotBrowseComp is a PFN™ benchmark for search agents. Each question is in the format of
[BrowseComp](https://arxiv.org/abs/2504.12516) and is answered by searching a fixed snapshot of the web, not the
live web, so that results do not change over time and no search API is needed.

- Questions: [`pfnet/snapshot-browsecomp`](https://huggingface.co/datasets/pfnet/snapshot-browsecomp)
  (Japanese, 250 questions)
- Search corpus: [`pfnet/snapshot-browsecomp-search-corpus`](https://huggingface.co/datasets/pfnet/snapshot-browsecomp-search-corpus)
  (Japanese and English, published as keys into public HF datasets)

See the [blog post](https://tech.preferred.jp/ja/blog/snapshot-browsecomp-ja) (in Japanese) for how the benchmark was made.

This repository has the scripts to rebuild the search corpus, index it and evaluate a model on it.
[`examples/ja.jsonl`](./examples/ja.jsonl) has a few questions made the same way as the benchmark's, in plain text.

```sh
pip install -r requirements.txt
```

## 1. Reconstruct the search corpus

The corpus is published as keys into public HF datasets, not as text. Rebuild the text with
[`reconstruct.py`](./reconstruct.py), for both languages (the model can search either corpus whichever language
it is evaluated on):

```sh
python reconstruct.py --lang ja --output-dir ./corpus-ja
python reconstruct.py --lang en --output-dir ./corpus-en
```

## 2. Start OpenSearch and build the index

[`docker-compose.yml`](./docker-compose.yml) runs a single-node OpenSearch on `localhost:9200`, with the
`analysis-kuromoji` and `analysis-icu` plugins the Japanese index needs. The data directory must be writable by
uid 1000, which OpenSearch runs as.

```sh
mkdir -p /path/to/disk/opensearch-data && sudo chown 1000:1000 /path/to/disk/opensearch-data
OPENSEARCH_DATA_DIR=/path/to/disk/opensearch-data OPENSEARCH_HEAP=16g docker compose up -d --build
```

[`build_index.py`](./build_index.py) indexes a reconstructed corpus into `<lang>-web-search-corpus`:

```sh
python build_index.py --lang ja --corpus-dir ./corpus-ja
python build_index.py --lang en --corpus-dir ./corpus-en
```

The corpora have 8.2M (ja) and 45.5M (en) documents. The Japanese index also stores kuromoji and character bigram
sub-fields with term vectors, so expect it to be well over the size of its text (40 GB).
`build_index.py` can be run again after an interruption: documents are indexed under fixed ids, so indexed ones are
overwritten rather than duplicated.

## 3. Evaluate

[`evaluate.py`](./evaluate.py) answers each question with the model in a tool-calling loop (`search(query, lang)`
over the `en` or `ja` corpus and `visit` to read a result), then has a judge model (`google/gemma-4-31b-it`
by default) compare the answer with the gold answer. Both models are called through OpenAI-compatible APIs.

```sh
python evaluate.py \
    --model <model> --base-url <url> --api-key-env <ENV holding the key> \
    --judge-base-url <url serving gemma-4-31b-it> --judge-api-key-env <ENV> \
    --output-dir ./results
```

- The model chooses which corpus each search goes to, so build both indexes even though the questions are
  Japanese only.
- `--questions` takes a HF dataset id (each language a subset) or a local file whose path may contain `{lang}`,
  e.g. `--questions './examples/{lang}.jsonl'`. A row has `problem`, `answer` and optionally `id`,
  `problem_topic` and `canary`; with a `canary`, `problem` and `answer` are encrypted as in BrowseComp.
- `--max-tool-calls` (default 40) bounds `search` and `visit` calls together; when it is spent, the model is asked
  for its answer without tools.
- `--difficulty lite` / `--difficulty hard` evaluates the questions of that difficulty only (both by default).
- `--no-tools` asks the model for the answer without the search tools, from its own knowledge only.
- Searches are BM25 over `title` and `text`, top `--max-results` (default 10) documents.
- `--opensearch-url` / `--index` point at another OpenSearch, and `--aoss-host` at an Amazon OpenSearch
  Serverless collection (with boto3 credentials).

`<output-dir>/<lang>.jsonl` has one record per question: the predicted answer, the judge's verdict and
explanation, and the whole conversation. `<output-dir>/summary.json` has the accuracy of each language and of all
of them. Running the same command again asks only the questions that have no record yet or whose record has an
`error`.

## License

The scripts and the questions are under the [Apache License 2.0](./LICENSE). The search corpus follows the
licenses of the datasets it points into; see its
[dataset card](https://huggingface.co/datasets/pfnet/snapshot-browsecomp-search-corpus).

## Citation

```bibtex
@misc{snapshotbrowsecomp2026,
  title        = {SnapshotBrowseComp},
  author       = {{Preferred Networks}},
  year         = {2026},
  howpublished = {\url{https://github.com/pfnet-research/snapshot-browsecomp}},
}
```
