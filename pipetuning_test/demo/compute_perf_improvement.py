#!/usr/bin/env python3
"""Compute per-row optimal-vs-overall performance improvements for a CSV file."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple

DEFAULT_CSV = Path(__file__).with_name("topk_underperforming_L20.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the per-row improvement ratio (optimal_perf / overall_perf) "
            "and the dataset-wide average for a tuning CSV."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to the CSV file to analyze (default: {DEFAULT_CSV.name}).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Optional path to write a new CSV with an added 'relative_improvement' column. "
            "If omitted, the script only prints the summary."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print the average improvement instead of all per-row entries.",
    )
    return parser.parse_args()


def safe_float(value: str, field_name: str, row_idx: int) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        raise ValueError(f"Row {row_idx}: failed to parse '{field_name}' as float (value={value!r}).")


def load_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV file contains no data rows: {csv_path}")
        return rows, reader.fieldnames or []


def compute_improvements(rows: List[Dict[str, str]]) -> List[float]:
    improvements: List[float] = []
    for idx, row in enumerate(rows, start=1):
        overall_perf = safe_float(row.get("overall_perf", ""), "overall_perf", idx)
        optimal_perf = safe_float(row.get("optimal_perf", ""), "optimal_perf", idx)

        if overall_perf == 0:
            raise ZeroDivisionError(f"Row {idx}: overall_perf is zero, cannot compute ratio.")

        ratio = optimal_perf / overall_perf
        row["relative_improvement"] = f"{ratio:.8f}"
        improvements.append(ratio)
    return improvements


def maybe_write_output(rows: List[Dict[str, str]], headers: List[str], output_path: Path | None) -> None:
    if output_path is None:
        return

    fieldnames = list(headers)
    if "relative_improvement" not in fieldnames:
        fieldnames.append("relative_improvement")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    try:
        rows, headers = load_rows(args.input)
        improvements = compute_improvements(rows)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    average_improvement = sum(improvements) / len(improvements)

    if not args.summary_only:
        print("weight_type\toptimal_perf\toverall_perf\trelative_improvement")
        for row, ratio in zip(rows, improvements, strict=True):
            weight_type = row.get("weight_type", "")
            opt = row.get("optimal_perf", "")
            overall = row.get("overall_perf", "")
            print(f"{weight_type}\t{opt}\t{overall}\t{ratio:.8f}")
        print()

    print(f"Average relative improvement (optimal/overall): {average_improvement:.8f}")

    try:
        maybe_write_output(rows, headers, args.output)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: failed to write output CSV: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
