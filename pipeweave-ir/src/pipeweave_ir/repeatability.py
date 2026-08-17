from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .benchmark import summarize_samples


def _load_run(run_root: Path) -> dict[str, Any]:
    run_manifest = json.loads((run_root / "run_manifest.json").read_text())
    cases: dict[str, dict[str, Any]] = {}
    for record in run_manifest.get("records", []):
        case_id = record.get("case_id")
        if not case_id:
            continue
        artifact_path = run_root / "artifacts" / case_id / "manifest.json"
        measurement_path = run_root / "measurements" / f"{case_id}.json"
        if not artifact_path.exists() or not measurement_path.exists():
            continue
        cases[case_id] = {
            "record": record,
            "artifact": json.loads(artifact_path.read_text()),
            "measurement": json.loads(measurement_path.read_text()),
        }
    return {
        "root": str(run_root.resolve()),
        "manifest": run_manifest,
        "cases": cases,
    }


def _artifact_consistency(runs: list[dict[str, Any]], case_ids: list[str]) -> dict[str, Any]:
    hashes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for case_id in case_ids:
        for run in runs:
            stages = run["cases"][case_id]["artifact"].get("stages", {})
            for stage, record in stages.items():
                hashes[stage][case_id].append(str(record.get("sha256")))

    result: dict[str, Any] = {}
    for stage, stage_cases in sorted(hashes.items()):
        comparable = {
            case_id: values for case_id, values in stage_cases.items() if len(values) == len(runs)
        }
        unstable = {
            case_id: values for case_id, values in comparable.items() if len(set(values)) != 1
        }
        result[stage] = {
            "comparable_cases": len(comparable),
            "stable_cases": len(comparable) - len(unstable),
            "stability_rate": (
                (len(comparable) - len(unstable)) / len(comparable) if comparable else 0.0
            ),
            "unstable": unstable,
        }
    return result


def _timing_repeatability(runs: list[dict[str, Any]], case_ids: list[str]) -> dict[str, Any]:
    samples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for case_id in case_ids:
        for run in runs:
            measurement = run["cases"][case_id]["measurement"]
            for timing in measurement.get("measurements", []):
                samples[timing["mode"]][case_id].append(timing)

    modes: dict[str, Any] = {}
    for mode, mode_cases in sorted(samples.items()):
        per_case: list[dict[str, Any]] = []
        for case_id, timings in sorted(mode_cases.items()):
            if len(timings) != len(runs):
                continue
            medians = [float(timing["summary"]["median"]) for timing in timings]
            cross_run = summarize_samples(medians)
            per_case.append(
                {
                    "case_id": case_id,
                    "case_name": runs[0]["cases"][case_id]["record"].get("case_name"),
                    "run_medians_us": medians,
                    "cross_run": cross_run,
                    "within_run_cv": [float(timing["summary"]["cv"]) for timing in timings],
                }
            )
        cross_run_cvs = [record["cross_run"]["cv"] for record in per_case]
        within_run_cvs = [value for record in per_case for value in record["within_run_cv"]]
        modes[mode] = {
            "comparable_cases": len(per_case),
            "median_cross_run_cv": statistics.median(cross_run_cvs) if cross_run_cvs else None,
            "p90_cross_run_cv": (
                summarize_samples(cross_run_cvs)["p90"] if cross_run_cvs else None
            ),
            "max_cross_run_cv": max(cross_run_cvs) if cross_run_cvs else None,
            "cases_cross_run_cv_over_3pct": sum(value > 0.03 for value in cross_run_cvs),
            "median_within_run_cv": statistics.median(within_run_cvs) if within_run_cvs else None,
            "per_case": per_case,
        }
    return modes


def compare_runs(run_roots: list[str | Path]) -> dict[str, Any]:
    if len(run_roots) < 2:
        raise ValueError("repeatability comparison requires at least two runs")
    runs = [_load_run(Path(root)) for root in run_roots]
    case_sets = [set(run["cases"]) for run in runs]
    common = sorted(set.intersection(*case_sets))
    union = sorted(set.union(*case_sets))
    return {
        "run_count": len(runs),
        "runs": [run["root"] for run in runs],
        "case_counts": [len(run["cases"]) for run in runs],
        "common_case_count": len(common),
        "union_case_count": len(union),
        "missing_by_run": {run["root"]: sorted(set(union) - set(run["cases"])) for run in runs},
        "artifact_consistency": _artifact_consistency(runs, common),
        "timing_repeatability": _timing_repeatability(runs, common),
    }
