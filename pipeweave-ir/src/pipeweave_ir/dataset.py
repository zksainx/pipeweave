from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import flatten_numeric
from .util import SCHEMA_VERSION, atomic_write_text, stable_hash, utc_now


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        records.append(value)
    return records


def _categorical(flat: dict[str, float], prefix: str, value: Any) -> None:
    if value is not None:
        flat[f"{prefix}.{value}"] = 1.0


def _base_features(manifest: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    flatten_numeric("base.problem", manifest.get("problem", {}), flat)
    flatten_numeric("base.config", manifest.get("config", {}), flat)
    flatten_numeric("base.launch", manifest.get("launch", {}), flat)

    identity = manifest.get("identity", {})
    _categorical(flat, "base.kernel", identity.get("kernel"))
    _categorical(flat, "base.family", identity.get("family"))
    _categorical(flat, "base.dtype", manifest.get("problem", {}).get("dtype"))

    target = manifest.get("target", {})
    _categorical(flat, "base.target.backend", target.get("backend"))
    flatten_numeric("base.target", target, flat)

    gpu = manifest.get("environment", {}).get("gpu", {})
    flatten_numeric("base.hardware", gpu, flat)
    return flat


def _compact_ir_features(feature_record: dict[str, Any]) -> dict[str, float]:
    features = feature_record.get("features", {})
    compact: dict[str, Any] = {}
    for stage in ("ttir", "ttgir", "llir", "ptx", "sass"):
        values = features.get(stage, {})
        if not isinstance(values, dict):
            continue
        summary = {
            key: values[key]
            for key in ("line_count", "operation_count", "instruction_count")
            if key in values
        }
        summary["categories"] = values.get("categories", {})
        summary["operation_kind_count"] = len(values.get("operations", {}))
        summary["instruction_kind_count"] = len(values.get("instructions", {}))
        summary["layout_kind_count"] = len(values.get("layouts", {}))
        compact[stage] = summary

    compiler = features.get("compiler", {})
    if isinstance(compiler, dict):
        compact["compiler"] = {
            key: compiler.get(key, 0)
            for key in (
                "global_scratch_size",
                "num_ctas",
                "num_stages",
                "num_warps",
                "profile_scratch_size",
                "shared",
                "tmem_size",
                "warp_size",
            )
        }
    compact["binary"] = features.get("binary", {}).get("maxima", {})
    compact["artifact_sizes"] = features.get("artifact_sizes", {})
    flat: dict[str, float] = {}
    flatten_numeric("compact", compact, flat)
    return flat


def _compiler_id(manifest: dict[str, Any]) -> str:
    environment = manifest.get("environment", {})
    compiler = {
        "triton": environment.get("triton", {}).get("version"),
        "torch": environment.get("torch", {}).get("version"),
        "cuda_runtime": environment.get("torch", {}).get("cuda_runtime"),
        "target": manifest.get("target", {}),
    }
    return stable_hash(compiler, length=20)


def _architecture_id(manifest: dict[str, Any]) -> str:
    target = manifest.get("target", {})
    backend = target.get("backend", "unknown")
    arch = target.get("arch", "unknown")
    return f"{backend}-{arch}"


def _dataset_row(
    manifest: dict[str, Any],
    feature_record: dict[str, Any],
    measurement: dict[str, Any],
    timing: dict[str, Any],
) -> dict[str, Any]:
    identity = manifest["identity"]
    problem = manifest.get("problem", {})
    config = manifest.get("config", {})
    flat = _base_features(manifest)
    for key, value in feature_record.get("flat_features", {}).items():
        flat[str(key)] = float(value)
    flat.update(_compact_ir_features(feature_record))

    kernel = identity.get("kernel")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "case_id": identity["case_id"],
        "case_name": identity.get("name"),
        "kernel": kernel,
        "family": identity.get("family"),
        "problem": problem,
        "config": config,
        "launch": manifest.get("launch", {}),
        "problem_group_id": stable_hash({"kernel": kernel, "problem": problem}, length=20),
        "config_group_id": stable_hash({"kernel": kernel, "config": config}, length=20),
        "kernel_source_id": identity.get("source_sha256"),
        "compiler_id": _compiler_id(manifest),
        "architecture_id": _architecture_id(manifest),
        "measurement_mode": timing["mode"],
        "label": {
            "latency_us": float(timing["summary"]["median"]),
            "unit": timing.get("unit", "us"),
            "summary": timing.get("summary", {}),
            "raw_sample_count": len(timing.get("raw_samples", [])),
        },
        "correctness": measurement.get("correctness", {}),
        "flat_features": dict(sorted(flat.items())),
    }


def build_dataset(
    run_root: str | Path,
    *,
    features_path: str | Path | None = None,
    modes: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_root = Path(run_root)
    if features_path is None:
        features_path = run_root / "features.jsonl"
    feature_records = read_jsonl(features_path)
    features_by_id = {record["case_id"]: record for record in feature_records}
    run_manifest = json.loads((run_root / "run_manifest.json").read_text())

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for record in run_manifest.get("records", []):
        case_id = record.get("case_id")
        if record.get("status") != "ok" or not case_id:
            skipped.append(
                {"case_name": str(record.get("case_name")), "reason": str(record.get("status"))}
            )
            continue
        feature_record = features_by_id.get(case_id)
        if feature_record is None:
            skipped.append(
                {"case_name": str(record.get("case_name")), "reason": "features_missing"}
            )
            continue

        artifact_path = run_root / "artifacts" / case_id / "manifest.json"
        measurement_path = run_root / "measurements" / f"{case_id}.json"
        if not artifact_path.exists() or not measurement_path.exists():
            skipped.append({"case_name": str(record.get("case_name")), "reason": "record_missing"})
            continue
        manifest = json.loads(artifact_path.read_text())
        measurement = json.loads(measurement_path.read_text())
        if not measurement.get("correctness", {}).get("passed", False):
            skipped.append(
                {"case_name": str(record.get("case_name")), "reason": "correctness_failed"}
            )
            continue
        for timing in measurement.get("measurements", []):
            if modes is not None and timing.get("mode") not in modes:
                continue
            rows.append(_dataset_row(manifest, feature_record, measurement, timing))

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(run_root.resolve()),
        "feature_records": len(feature_records),
        "dataset_rows": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "measurement_modes": sorted({row["measurement_mode"] for row in rows}),
        "skipped": skipped,
    }
    return rows, summary


def write_dataset(path: str | Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    atomic_write_text(Path(path), "\n".join(lines) + ("\n" if lines else ""))
