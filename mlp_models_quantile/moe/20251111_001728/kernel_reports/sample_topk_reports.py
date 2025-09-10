#!/usr/bin/env python3
"""Sample topk_* kernel report CSVs to cap M combinations per shape."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_GROUP_COLUMNS = ("E", "topk", "H", "N")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce topk_* CSVs by keeping at most N rows per (E, topk, H, N) \n"
            "combination (i.e., at most N different M values per shape)."
        )
    )
    parser.add_argument(
        "reports_dir",
        nargs="?",
        default=".",
        help="Directory that contains the topk CSV files (default: current directory).",
    )
    parser.add_argument(
        "--prefix",
        default="topk",
        help="Only files whose name starts with this prefix are processed (default: topk).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Maximum number of rows kept per (E, topk, H, N) combination (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for randomness when sampling rows (default: 0).",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=list(DEFAULT_GROUP_COLUMNS),
        help=(
            "Columns that define a combination before sampling. "
            "By default only (E, topk, H, N) are used so that rows differing "
            "in M are considered duplicates."
        ),
    )
    parser.add_argument(
        "--sort-column",
        default="perf_diff",
        help="Column used to sort the reduced CSV ascending (default: perf_diff).",
    )
    return parser.parse_args()

def discover_csvs(directory: Path, prefix: str) -> List[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix == ".csv" and p.name.startswith(prefix)
    )

def sample_rows(rows: List[dict], group_columns: Iterable[str], sample_size: int, rng: random.Random) -> List[dict]:
    groups: defaultdict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in rows:
        try:
            key = tuple(row[col] for col in group_columns)
        except KeyError as exc:  # pragma: no cover - defensive guard
            missing = exc.args[0]
            raise KeyError(f"Column '{missing}' not in CSV header") from exc
        groups[key].append(row)

    sampled: List[dict] = []
    for entries in groups.values():
        if sample_size < len(entries):
            sampled.extend(rng.sample(entries, sample_size))
        else:
            sampled.extend(entries)
    return sampled

def sort_rows(rows: List[dict], sort_column: str) -> None:
    def sort_key(row: dict) -> float:
        value = row.get(sort_column)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    rows.sort(key=sort_key)

def process_file(path: Path, sample_size: int, group_columns: Iterable[str], sort_column: str, rng: random.Random) -> Tuple[Path, int, int]:
    with path.open(newline="") as src:
        reader = csv.DictReader(src)
        rows = list(reader)
        header = reader.fieldnames

    if not rows:
        return path, 0, 0
    if sample_size <= 0:
        raise ValueError("sample-size must be positive")
    sampled = sample_rows(rows, group_columns, sample_size, rng)
    sort_rows(sampled, sort_column)

    output_path = path.with_name(f"{path.stem}_sampled{path.suffix}")
    with output_path.open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=header)
        writer.writeheader()
        writer.writerows(sampled)

    return output_path, len(rows), len(sampled)

def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_dir():
        raise SystemExit(f"Directory not found: {reports_dir}")

    csv_files = discover_csvs(reports_dir, args.prefix)
    if not csv_files:
        raise SystemExit(f"No CSVs starting with '{args.prefix}' found in {reports_dir}")

    rng = random.Random(args.seed)
    for csv_path in csv_files:
        output_path, total, kept = process_file(
            csv_path,
            args.sample_size,
            args.group_columns,
            args.sort_column,
            rng,
        )
        print(f"{csv_path.name}: kept {kept}/{total} rows -> {output_path.name}")

if __name__ == "__main__":
    main()
