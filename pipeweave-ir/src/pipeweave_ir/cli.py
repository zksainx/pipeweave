from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ablation import run_ablation
from .capture import capture_suite
from .dataset import build_dataset, read_jsonl, write_dataset
from .environment import collect_environment
from .features import extract_artifact_features
from .repeatability import compare_runs
from .suitegen import build_rtx5090_day6_suite
from .util import atomic_write_json, atomic_write_text


def _environment_command(args: argparse.Namespace) -> int:
    environment = collect_environment(
        require_cuda_major=args.require_cuda_major,
        cuda_bin=args.cuda_bin,
    )
    atomic_write_json(args.out, environment)
    print(json.dumps(environment, indent=2, sort_keys=True))
    return 0


def _capture_command(args: argparse.Namespace) -> int:
    result = capture_suite(
        args.suite,
        output_root=args.out,
        require_cuda_major=args.require_cuda_major,
        cuda_bin=args.cuda_bin,
        limit=args.limit,
        measure=not args.skip_measure,
        enable_graph=not args.no_graph,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["summary"]["error"] == 0 else 1


def _feature_command(args: argparse.Namespace) -> int:
    artifact_root = Path(args.artifacts)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for manifest in sorted(artifact_root.glob("*/manifest.json")):
        try:
            records.append(extract_artifact_features(manifest.parent))
        except Exception as exc:
            errors.append(
                {
                    "manifest": str(manifest),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    lines = [json.dumps(record, sort_keys=True) for record in records]
    atomic_write_text(Path(args.out), "\n".join(lines) + ("\n" if lines else ""))
    summary = {"records": len(records), "errors": errors}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _dataset_command(args: argparse.Namespace) -> int:
    modes = set(args.mode) if args.mode else None
    rows, summary = build_dataset(args.run, features_path=args.features, modes=modes)
    write_dataset(args.out, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rows else 1


def _ablation_command(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.dataset)
    result = run_ablation(
        rows,
        measurement_mode=args.mode,
        split_by=args.split_by,
        test_fraction=args.test_fraction,
        seed=args.seed,
        alpha=args.alpha,
        family=args.family,
    )
    atomic_write_json(args.out, result)
    summary = {name: values["test_metrics"] for name, values in result["ablations"].items()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _generate_command(args: argparse.Namespace) -> int:
    builders = {"rtx5090-day6": build_rtx5090_day6_suite}
    suite = builders[args.profile]()
    atomic_write_json(args.out, suite)
    print(json.dumps({"suite": suite["suite"], "cases": len(suite["cases"])}, indent=2))
    return 0


def _repeatability_command(args: argparse.Namespace) -> int:
    result = compare_runs(args.run)
    atomic_write_json(args.out, result)
    summary = {
        "run_count": result["run_count"],
        "common_case_count": result["common_case_count"],
        "artifact_consistency": {
            stage: values["stability_rate"]
            for stage, values in result["artifact_consistency"].items()
        },
        "timing_repeatability": {
            mode: {
                "median_cross_run_cv": values["median_cross_run_cv"],
                "p90_cross_run_cv": values["p90_cross_run_cv"],
                "max_cross_run_cv": values["max_cross_run_cv"],
                "cases_cross_run_cv_over_3pct": values["cases_cross_run_cv_over_3pct"],
            }
            for mode, values in result["timing_repeatability"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeweave-ir")
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment = subparsers.add_parser("env", help="capture the toolchain and GPU environment")
    environment.add_argument("--out", type=Path, required=True)
    environment.add_argument("--require-cuda-major", type=int)
    environment.add_argument("--cuda-bin", type=Path)
    environment.set_defaults(handler=_environment_command)

    capture = subparsers.add_parser("capture", help="compile, inspect, and benchmark a suite")
    capture.add_argument("--suite", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--require-cuda-major", type=int)
    capture.add_argument("--cuda-bin", type=Path)
    capture.add_argument("--limit", type=int, default=0)
    capture.add_argument("--skip-measure", action="store_true")
    capture.add_argument("--no-graph", action="store_true")
    capture.set_defaults(handler=_capture_command)

    features = subparsers.add_parser("features", help="extract static features from artifacts")
    features.add_argument("--artifacts", type=Path, required=True)
    features.add_argument("--out", type=Path, required=True)
    features.set_defaults(handler=_feature_command)

    dataset = subparsers.add_parser("dataset", help="join artifacts, features, and timing labels")
    dataset.add_argument("--run", type=Path, required=True)
    dataset.add_argument("--features", type=Path)
    dataset.add_argument("--out", type=Path, required=True)
    dataset.add_argument("--mode", action="append")
    dataset.set_defaults(handler=_dataset_command)

    ablation = subparsers.add_parser("ablate", help="run grouped ridge feature ablations")
    ablation.add_argument("--dataset", type=Path, required=True)
    ablation.add_argument("--out", type=Path, required=True)
    ablation.add_argument("--mode", default="eager_event")
    ablation.add_argument("--family")
    ablation.add_argument(
        "--split-by",
        choices=("shape", "config", "compiler", "architecture"),
        default="shape",
    )
    ablation.add_argument("--test-fraction", type=float, default=0.25)
    ablation.add_argument("--seed", type=int, default=17)
    ablation.add_argument("--alpha", type=float, default=1.0)
    ablation.set_defaults(handler=_ablation_command)

    generate = subparsers.add_parser("generate", help="generate a reproducible case matrix")
    generate.add_argument("--profile", choices=("rtx5090-day6",), required=True)
    generate.add_argument("--out", type=Path, required=True)
    generate.set_defaults(handler=_generate_command)

    repeatability = subparsers.add_parser(
        "repeatability", help="compare compiler hashes and timing across runs"
    )
    repeatability.add_argument("--run", type=Path, action="append", required=True)
    repeatability.add_argument("--out", type=Path, required=True)
    repeatability.set_defaults(handler=_repeatability_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
