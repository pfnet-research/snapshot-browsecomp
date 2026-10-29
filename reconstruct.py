"""Reconstructs the snapshot_browsecomp search corpus from its keys and the source HF datasets.

Each key points to a row (``hf_repo``, ``hf_revision``, ``file``, ``row``) of a source dataset. The text is read
from there; ``category`` and ``year`` come from the key. The output has one parquet file per source file, with the
columns ``title``, ``text``, ``url``, ``category`` and ``year``, ordered by source file and row.

This file only depends on pyarrow, datasets and huggingface_hub, so that it can be distributed with the keys.

    python reconstruct.py --lang ja --output-dir ./corpus-ja
"""

import argparse
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

KEYS_DATASET = "pfnet/snapshot-browsecomp-search-corpus"

_KEY_COLUMNS = ["source", "hf_repo", "hf_revision", "file", "row", "id", "category", "year"]


@dataclass(frozen=True)
class _Source:
    columns: list[str]
    title: Callable[[dict[str, Any]], str]
    url: Callable[[dict[str, Any]], str]


def _empty_title(row: dict[str, Any]) -> str:
    return ""


_SOURCES = {
    "fineweb": _Source(["id", "text", "url"], _empty_title, lambda r: r["url"]),
    "fineweb-2": _Source(["id", "text", "url"], _empty_title, lambda r: r["url"]),
    "fineweb-edu-dedup": _Source(["id", "text", "metadata"], _empty_title, lambda r: r["metadata"]["url"]),
    "finewiki": _Source(["id", "text", "url", "title"], lambda r: r["title"], lambda r: r["url"]),
}


@dataclass(frozen=True)
class FileKeys:
    """The keys that point to one source file."""

    source: str
    hf_repo: str
    hf_revision: str
    file: str
    rows: list[int]
    ids: list[str]
    categories: list[str]
    years: list[int]


def group_keys(keys: pa.Table) -> list[FileKeys]:
    grouped = keys.select(_KEY_COLUMNS).group_by(["source", "hf_repo", "hf_revision", "file"], use_threads=False)
    table = grouped.aggregate([("row", "list"), ("id", "list"), ("category", "list"), ("year", "list")])
    groups = [
        FileKeys(
            source=g["source"],
            hf_repo=g["hf_repo"],
            hf_revision=g["hf_revision"],
            file=g["file"],
            rows=g["row_list"],
            ids=g["id_list"],
            categories=g["category_list"],
            years=g["year_list"],
        )
        for g in table.to_pylist()
    ]
    return sorted(groups, key=lambda g: (g.hf_repo, g.file))


def _open_hf_file(repo: str, revision: str, path: str) -> BinaryIO:
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(f"datasets/{repo}@{revision}/{path}", "rb")


def output_path(output_dir: str, keys: FileKeys) -> str:
    return os.path.join(output_dir, keys.source, keys.file.replace("/", "__"))


def reconstruct_file(
    keys: FileKeys, output_dir: str, open_fn: Callable[[str, str, str], BinaryIO] = _open_hf_file
) -> str:
    """Writes the documents of ``keys`` to ``output_path(output_dir, keys)``. Existing outputs are kept as is."""
    out = output_path(output_dir, keys)
    if os.path.exists(out):
        return out
    source = _SOURCES[keys.source]
    order = sorted(range(len(keys.rows)), key=lambda i: keys.rows[i])
    records: list[dict[str, Any]] = []
    with open_fn(keys.hf_repo, keys.hf_revision, keys.file) as f:
        pf = pq.ParquetFile(f)
        group_start = 0
        cursor = 0
        for group in range(pf.metadata.num_row_groups):
            group_end = group_start + pf.metadata.row_group(group).num_rows
            needed = []
            while cursor < len(order) and keys.rows[order[cursor]] < group_end:
                needed.append(order[cursor])
                cursor += 1
            if needed:
                table = pf.read_row_group(group, columns=source.columns)
                rows = table.take([keys.rows[i] - group_start for i in needed]).to_pylist()
                for i, row in zip(needed, rows):
                    if row["id"] != keys.ids[i]:
                        raise ValueError(f"{keys.file} row {keys.rows[i]} has id {row['id']}, expected {keys.ids[i]}")
                    records.append(
                        {
                            "title": source.title(row),
                            "text": row["text"],
                            "url": source.url(row),
                            "category": keys.categories[i],
                            "year": keys.years[i],
                        }
                    )
            group_start = group_end
        if cursor != len(order):
            raise ValueError(f"{keys.file} has only {group_start} rows")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = f"{out}.tmp"
    pq.write_table(pa.Table.from_pylist(records), tmp)
    os.replace(tmp, out)
    return out


def load_keys(keys: str, lang: str) -> pa.Table:
    import datasets

    if os.path.exists(keys):
        files = [os.path.join(keys, f) for f in sorted(os.listdir(keys))] if os.path.isdir(keys) else [keys]
        files = [f for f in files if f.endswith((".parquet", ".jsonl", ".jsonl.zst"))]
        builder = "parquet" if files[0].endswith(".parquet") else "json"
        ds = datasets.load_dataset(builder, data_files=files, split="train")
    else:
        ds = datasets.load_dataset(keys, lang, split="train")
    return ds.select_columns(_KEY_COLUMNS).data.table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", type=str, required=True, choices=["ja", "en"])
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--keys", type=str, default=KEYS_DATASET, help="HF dataset id, or a local file / dir of keys")
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    groups = group_keys(load_keys(args.keys, args.lang))
    print(f"{sum(len(g.rows) for g in groups)} documents in {len(groups)} source files")
    with ProcessPoolExecutor(args.num_proc) as ex:
        for i, out in enumerate(ex.map(reconstruct_file, groups, [args.output_dir] * len(groups))):
            print(f"[{i + 1}/{len(groups)}] {out}", flush=True)


if __name__ == "__main__":
    main()
