from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from .metrics import latency_metrics, ranking_metrics
from .util import stable_hash

ABLATIONS: dict[str, tuple[str, ...]] = {
    "shape_config": ("base",),
    "ttir": ("base", "ttir"),
    "ttgir": ("base", "ttgir"),
    "ptx": ("base", "ptx"),
    "sass_binary": ("base", "sass", "binary", "compiler"),
    "ttgir_ptx": ("base", "ttgir", "ptx"),
    "compact_ttgir": ("base", "compact.ttgir"),
    "compact_ptx": ("base", "compact.ptx"),
    "compact_ttgir_ptx": ("base", "compact.ttgir", "compact.ptx"),
    "compact_combined": (
        "base",
        "compact.ttgir",
        "compact.ptx",
        "compact.sass",
        "compact.binary",
        "compact.compiler",
        "compact.artifact_sizes",
    ),
    "combined": (
        "base",
        "ttir",
        "ttgir",
        "llir",
        "ptx",
        "sass",
        "binary",
        "compiler",
        "artifact_sizes",
    ),
}


def deterministic_group_split(
    rows: list[dict[str, Any]],
    *,
    group_field: str,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one")
    groups = sorted({str(row[group_field]) for row in rows})
    if len(groups) < 2:
        raise ValueError(f"at least two {group_field} groups are required")
    ordered = sorted(groups, key=lambda group: stable_hash({"group": group, "seed": seed}))
    test_count = min(len(groups) - 1, max(1, round(len(groups) * test_fraction)))
    test_groups = set(ordered[:test_count])
    train = [row for row in rows if str(row[group_field]) not in test_groups]
    test = [row for row in rows if str(row[group_field]) in test_groups]
    return (
        train,
        test,
        {
            "group_field": group_field,
            "seed": seed,
            "test_fraction": test_fraction,
            "train_groups": sorted(set(groups) - test_groups),
            "test_groups": sorted(test_groups),
            "train_rows": len(train),
            "test_rows": len(test),
        },
    )


def _matches_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)


def _feature_keys(rows: list[dict[str, Any]], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(
        {key for row in rows for key in row["flat_features"] if _matches_prefix(key, prefixes)}
    )


def _fit_predict_ridge(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    *,
    prefixes: tuple[str, ...],
    alpha: float,
) -> tuple[list[float], list[float], dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for the ridge ablation") from exc

    keys = _feature_keys(train, prefixes)
    if not keys:
        raise ValueError(f"no features found for prefixes {prefixes}")

    def matrix(rows: list[dict[str, Any]]) -> Any:
        return np.asarray(
            [[float(row["flat_features"].get(key, 0.0)) for key in keys] for row in rows],
            dtype=np.float64,
        )

    x_train = matrix(train)
    x_test = matrix(test)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    keep = scale > 1e-12
    x_train = (x_train[:, keep] - mean[keep]) / scale[keep]
    x_test = (x_test[:, keep] - mean[keep]) / scale[keep]

    y_train = np.log(np.asarray([row["label"]["latency_us"] for row in train], dtype=np.float64))
    y_mean = float(y_train.mean())
    centered = y_train - y_mean
    feature_count = int(x_train.shape[1])
    if feature_count == 0:
        train_log = np.full(len(train), y_mean)
        test_log = np.full(len(test), y_mean)
    elif feature_count <= len(train):
        system = x_train.T @ x_train + alpha * np.eye(feature_count)
        weights = np.linalg.solve(system, x_train.T @ centered)
        train_log = x_train @ weights + y_mean
        test_log = x_test @ weights + y_mean
    else:
        system = x_train @ x_train.T + alpha * np.eye(len(train))
        dual = np.linalg.solve(system, centered)
        weights = x_train.T @ dual
        train_log = x_train @ weights + y_mean
        test_log = x_test @ weights + y_mean

    train_predictions = np.exp(np.clip(train_log, -20.0, 20.0)).tolist()
    test_predictions = np.exp(np.clip(test_log, -20.0, 20.0)).tolist()
    metadata = {
        "candidate_feature_count": len(keys),
        "retained_feature_count": feature_count,
        "alpha": alpha,
        "target_transform": "log",
    }
    return train_predictions, test_predictions, metadata


def _ranking_summary(rows: list[dict[str, Any]], predictions: list[float]) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row, prediction in zip(rows, predictions, strict=True):
        grouped[row["problem_group_id"]].append((row["label"]["latency_us"], prediction))

    per_problem: list[dict[str, Any]] = []
    for group_id, values in sorted(grouped.items()):
        if len(values) < 2:
            continue
        actual = [value[0] for value in values]
        predicted = [value[1] for value in values]
        result = ranking_metrics(actual, predicted, top_k=min(3, len(values)))
        per_problem.append({"problem_group_id": group_id, **result})
    if not per_problem:
        return {"problem_count": 0, "per_problem": []}
    return {
        "problem_count": len(per_problem),
        "mean_spearman": statistics.fmean(row["spearman"] for row in per_problem),
        "mean_top_k_recall": statistics.fmean(row["top_k_recall"] for row in per_problem),
        "mean_regret_percent": statistics.fmean(row["regret_percent"] for row in per_problem),
        "per_problem": per_problem,
    }


def run_ablation(
    rows: list[dict[str, Any]],
    *,
    measurement_mode: str,
    split_by: str,
    test_fraction: float,
    seed: int,
    alpha: float,
    family: str | None = None,
) -> dict[str, Any]:
    selected = [row for row in rows if row["measurement_mode"] == measurement_mode]
    if family is not None:
        selected = [row for row in selected if row.get("family") == family]
    if len(selected) < 3:
        raise ValueError("at least three matching dataset rows are required")

    split_fields = {
        "shape": "problem_group_id",
        "config": "config_group_id",
        "compiler": "compiler_id",
        "architecture": "architecture_id",
    }
    try:
        group_field = split_fields[split_by]
    except KeyError as exc:
        raise ValueError(f"unsupported split: {split_by}") from exc
    train, test, split = deterministic_group_split(
        selected,
        group_field=group_field,
        test_fraction=test_fraction,
        seed=seed,
    )

    actual_train = [row["label"]["latency_us"] for row in train]
    actual_test = [row["label"]["latency_us"] for row in test]
    results: dict[str, Any] = {}
    for name, prefixes in ABLATIONS.items():
        train_predicted, test_predicted, model = _fit_predict_ridge(
            train,
            test,
            prefixes=prefixes,
            alpha=alpha,
        )
        results[name] = {
            "prefixes": list(prefixes),
            "model": model,
            "train_metrics": latency_metrics(actual_train, train_predicted),
            "test_metrics": latency_metrics(actual_test, test_predicted),
            "test_ranking": _ranking_summary(test, test_predicted),
            "test_predictions": [
                {
                    "case_id": row["case_id"],
                    "case_name": row["case_name"],
                    "problem_group_id": row["problem_group_id"],
                    "actual_us": actual,
                    "predicted_us": predicted,
                    "ape": abs(predicted - actual) / actual * 100.0,
                }
                for row, actual, predicted in zip(test, actual_test, test_predicted, strict=True)
            ],
        }
    return {
        "measurement_mode": measurement_mode,
        "family": family,
        "row_count": len(selected),
        "split": split,
        "ridge_alpha": alpha,
        "ablations": results,
        "caveat": (
            "This baseline is diagnostic. Statistical claims require more shapes, repeated runs, "
            "and an architecture holdout."
        ),
    }
