"""Builds the OpenSearch index of the snapshot_browsecomp search corpus for ``evaluate.py``.

The corpus is the output of ``reconstruct.py`` (parquet files with ``title``, ``text``, ``url``, ``category`` and
``year``). The index has the analyzers of the index the benchmark was made with, so that a local index ranks
documents as that one does: Japanese text is matched on kuromoji (``.analyzed``) and character bigram (``.ngram``)
sub-fields, English text on the standard analyzer. Kuromoji and the ICU normalizer need the ``analysis-kuromoji``
and ``analysis-icu`` plugins, which the ``Dockerfile`` next to this file installs.

A document's id is its file and row in the corpus, so running this again after an interruption overwrites what
was indexed rather than adding duplicates.

This file only depends on pyarrow and opensearch-py, so that it can be distributed with the benchmark.

    python build_index.py --lang ja --corpus-dir ./corpus-ja
"""

import argparse
import os
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq

DEFAULT_OPENSEARCH_URL = "http://localhost:9200"
# The index `evaluate.py` searches by default: `<lang>-web-search-corpus`.
DEFAULT_INDEX = "{lang}-web-search-corpus"

_FIELDS = ["title", "text", "url", "category", "year"]

_JA_ANALYSIS = {
    "char_filter": {"normalize": {"type": "icu_normalizer", "name": "nfkc", "mode": "compose"}},
    "tokenizer": {
        "ja_kuromoji_tokenizer": {"mode": "custom", "type": "kuromoji_tokenizer", "discard_compound_token": True},
        "ja_ngram_tokenizer": {"type": "ngram", "min_gram": 1, "max_gram": 2, "token_chars": ["letter", "digit"]},
    },
    "analyzer": {
        "ja_kuromoji_analyzer": {
            "type": "custom",
            "tokenizer": "ja_kuromoji_tokenizer",
            "char_filter": ["normalize"],
            "filter": [
                "kuromoji_baseform",
                "kuromoji_part_of_speech",
                "cjk_width",
                "ja_stop",
                "kuromoji_stemmer",
                "lowercase",
            ],
        },
        "ja_ngram_index_analyzer": {
            "type": "custom",
            "char_filter": ["normalize"],
            "tokenizer": "ja_ngram_tokenizer",
            "filter": ["lowercase"],
        },
    },
}


def _text_field(lang: str) -> dict[str, Any]:
    field: dict[str, Any] = {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}
    if lang == "ja":
        for name, analyzer in [("analyzed", "ja_kuromoji_analyzer"), ("ngram", "ja_ngram_index_analyzer")]:
            field["fields"][name] = {
                "type": "text",
                "analyzer": analyzer,
                "search_analyzer": analyzer,
                "term_vector": "with_positions_offsets",
            }
    return field


def index_body(lang: str, num_shards: int) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "number_of_shards": num_shards,
        "number_of_replicas": 0,
        # Turned back on by `build_index` once everything is indexed.
        "refresh_interval": "-1",
    }
    if lang == "ja":
        settings["analysis"] = _JA_ANALYSIS
    return {
        "settings": settings,
        "mappings": {
            "properties": {
                "title": _text_field(lang),
                "text": _text_field(lang),
                "url": {"type": "keyword", "ignore_above": 8191},
                "category": {"type": "keyword"},
                "year": {"type": "integer"},
            }
        },
    }


def corpus_files(corpus_dir: str) -> list[str]:
    """The parquet files under ``corpus_dir``, as paths relative to it, in a fixed order."""
    files = []
    for root, _, names in os.walk(corpus_dir):
        files.extend(os.path.relpath(os.path.join(root, n), corpus_dir) for n in names if n.endswith(".parquet"))
    return sorted(files)


def bulk_actions(corpus_dir: str, files: list[str], index: str, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    for file in files:
        pf = pq.ParquetFile(os.path.join(corpus_dir, file))
        row = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=_FIELDS):
            for doc in batch.to_pylist():
                yield {"_op_type": "index", "_index": index, "_id": f"{file}:{row}", "_source": doc}
                row += 1


def make_client(url: str) -> Any:
    from opensearchpy import OpenSearch

    return OpenSearch(hosts=[url], http_compress=True, timeout=600, retry_on_timeout=True, max_retries=5)


def build_index(
    client: Any,
    *,
    lang: str,
    corpus_dir: str,
    index: str,
    num_shards: int,
    num_threads: int,
    chunk_bytes: int,
    recreate: bool = False,
) -> int:
    from opensearchpy import helpers

    if recreate and client.indices.exists(index=index):
        client.indices.delete(index=index)
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=index_body(lang, num_shards))
    else:
        print(f"Index {index} already exists; indexing into it")
        client.indices.put_settings(index=index, body={"index": {"refresh_interval": "-1"}})

    files = corpus_files(corpus_dir)
    if not files:
        raise ValueError(f"No parquet files under {corpus_dir}")
    print(f"Indexing {len(files)} files from {corpus_dir} into {index}")
    n = 0
    for _ in helpers.parallel_bulk(
        client,
        bulk_actions(corpus_dir, files, index),
        thread_count=num_threads,
        chunk_size=10_000,
        max_chunk_bytes=chunk_bytes,
        raise_on_error=True,
    ):
        n += 1
        if n % 100_000 == 0:
            print(f"{n} documents indexed", flush=True)

    client.indices.put_settings(index=index, body={"index": {"refresh_interval": "1s"}})
    client.indices.refresh(index=index)
    count = client.count(index=index)["count"]
    print(f"Indexed {n} documents; {index} has {count} documents")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", type=str, required=True, choices=["ja", "en"])
    parser.add_argument("--corpus-dir", type=str, required=True, help="The --output-dir of reconstruct.py")
    parser.add_argument("--opensearch-url", type=str, default=DEFAULT_OPENSEARCH_URL)
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX, help="Index name; {lang} is replaced")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--chunk-bytes", type=int, default=16 * 1024 * 1024, help="Maximum size of a bulk request")
    parser.add_argument("--recreate", action="store_true", help="Delete the index first if it exists")
    args = parser.parse_args()

    build_index(
        make_client(args.opensearch_url),
        lang=args.lang,
        corpus_dir=args.corpus_dir,
        index=args.index.format(lang=args.lang),
        num_shards=args.num_shards,
        num_threads=args.num_threads,
        chunk_bytes=args.chunk_bytes,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
