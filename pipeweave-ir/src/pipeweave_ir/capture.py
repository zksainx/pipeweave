from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore
from .benchmark import (
    benchmark_cuda_event,
    benchmark_cuda_graph,
    benchmark_cuda_graph_batched,
    compare_tensors,
)
from .environment import collect_environment
from .nvidia import inspect_cubin
from .util import SCHEMA_VERSION, atomic_write_json, json_safe, stable_hash, utc_now


@dataclass
class PreparedCase:
    case: dict[str, Any]
    jit_function: Any
    arguments: tuple[Any, ...]
    launch_kwargs: dict[str, Any]
    grid: tuple[int, ...]
    output: Any
    expected: Any
    atol: float
    rtol: float

    def run(self) -> None:
        self.jit_function[self.grid](*self.arguments, **self.launch_kwargs)

    def compile(self) -> Any:
        return self.jit_function.run(
            *self.arguments,
            grid=self.grid,
            warmup=True,
            **self.launch_kwargs,
        )


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = json.loads(Path(path).read_text())
    if suite.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported suite schema: {suite.get('schema_version')!r}")
    if not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("suite must contain a non-empty cases list")
    names = [case.get("name") for case in suite["cases"]]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique within a suite")
    return suite


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _prepare_matmul(case: dict[str, Any]) -> PreparedCase:
    import torch
    import triton

    from .kernels import matmul_kernel, require_triton

    require_triton()
    problem = case["problem"]
    config = case["config"]
    m, n, k = int(problem["m"]), int(problem["n"]), int(problem["k"])
    dtype = _torch_dtype(problem["dtype"])
    generator = torch.Generator(device="cuda").manual_seed(17)
    a = torch.randn((m, k), device="cuda", dtype=dtype, generator=generator)
    b = torch.randn((k, n), device="cuda", dtype=dtype, generator=generator)
    output = torch.empty((m, n), device="cuda", dtype=dtype)
    expected = torch.matmul(a.float(), b.float())
    block_m = int(config["block_m"])
    block_n = int(config["block_n"])
    block_k = int(config["block_k"])
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    kwargs = {
        "M": m,
        "N": n,
        "K": k,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "num_warps": int(config["num_warps"]),
        "num_stages": int(config["num_stages"]),
    }
    return PreparedCase(
        case=case,
        jit_function=matmul_kernel,
        arguments=(a, b, output),
        launch_kwargs=kwargs,
        grid=grid,
        output=output,
        expected=expected,
        atol=0.08,
        rtol=0.04,
    )


def _prepare_attention(case: dict[str, Any]) -> PreparedCase:
    import torch
    import triton

    from .kernels import attention_kernel, require_triton

    require_triton()
    problem = case["problem"]
    config = case["config"]
    batch = int(problem["batch"])
    heads = int(problem["heads"])
    n_ctx = int(problem["n_ctx"])
    head_dim = int(problem["head_dim"])
    causal = bool(problem["causal"])
    dtype = _torch_dtype(problem["dtype"])
    generator = torch.Generator(device="cuda").manual_seed(23)
    shape = (batch, heads, n_ctx, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    k = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    output = torch.empty_like(q)
    scale = head_dim**-0.5

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        mask = torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool).triu(1)
        scores.masked_fill_(mask, float("-inf"))
    expected = torch.matmul(scores.softmax(dim=-1), v.float())

    block_m = int(config["block_m"])
    block_n = int(config["block_n"])
    grid = (triton.cdiv(n_ctx, block_m), batch * heads)
    kwargs = {
        "softmax_scale": scale,
        "BH": batch * heads,
        "N_CTX": n_ctx,
        "HEAD_DIM": head_dim,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "CAUSAL": causal,
        "num_warps": int(config["num_warps"]),
        "num_stages": int(config["num_stages"]),
    }
    return PreparedCase(
        case=case,
        jit_function=attention_kernel,
        arguments=(q, k, v, output),
        launch_kwargs=kwargs,
        grid=grid,
        output=output,
        expected=expected,
        atol=0.10,
        rtol=0.05,
    )


def prepare_case(case: dict[str, Any]) -> PreparedCase:
    kernel = case.get("kernel")
    if kernel == "matmul":
        return _prepare_matmul(case)
    if kernel == "attention":
        return _prepare_attention(case)
    raise ValueError(f"unknown kernel: {kernel!r}")


def _compiled_stages(compiled: Any) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for name in compiled.asm.keys():
        try:
            stages[str(name)] = compiled.asm[name]
        except Exception as exc:
            stages[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return stages


def _target_dict() -> dict[str, Any]:
    import triton

    target = triton.runtime.driver.active.get_current_target()
    return {"backend": target.backend, "arch": target.arch, "warp_size": target.warp_size}


def capture_case(
    case: dict[str, Any],
    *,
    suite_name: str,
    output_root: str | Path,
    environment: dict[str, Any],
    cuda_bin: str | Path | None = None,
    measure: bool = True,
    enable_graph: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_root)
    prepared = prepare_case(case)
    source = getattr(prepared.jit_function, "raw_src", repr(prepared.jit_function))
    target = _target_dict()
    source_hash = stable_hash(source)
    case_identity = {
        "suite": suite_name,
        "name": case["name"],
        "kernel": case["kernel"],
        "family": case.get("family"),
        "problem": case["problem"],
        "config": case["config"],
        "source_sha256": source_hash,
        "target": target,
    }
    case_id = stable_hash(case_identity, length=20)
    store = ArtifactStore(output_root / "artifacts")

    compiled = prepared.compile()
    stages = _compiled_stages(compiled)
    compiler_metadata = json_safe(getattr(compiled, "metadata", {}))
    stage_records = store.write_stages(case_id, stages)
    binary_inspection: dict[str, Any] = {"status": "not_available"}
    report_records: dict[str, Any] = {}
    if "cubin" in stage_records:
        cubin_path = store.case_dir(case_id) / stage_records["cubin"]["filename"]
        binary_inspection = inspect_cubin(cubin_path, cuda_bin=cuda_bin)
        for report_name, report in binary_inspection.get("reports", {}).items():
            text = f"{report.get('stdout', '')}\n{report.get('stderr', '')}"
            suffix = ".sass" if report_name == "nvdisasm" else ".txt"
            report_records[report_name] = store.write_report(case_id, report_name + suffix, text)

    launch = {
        "grid": list(prepared.grid),
        "num_warps": int(case["config"]["num_warps"]),
        "threads_per_cta": int(case["config"]["num_warps"]) * target["warp_size"],
        "num_stages": int(case["config"]["num_stages"]),
        "num_ctas": int(compiler_metadata.get("num_ctas", 1)),
        "dynamic_shared_bytes": int(compiler_metadata.get("shared", 0)),
        "global_scratch_bytes": int(compiler_metadata.get("global_scratch_size", 0)),
        "tmem_size": int(compiler_metadata.get("tmem_size", 0)),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "status": "ok",
        "identity": {**case_identity, "case_id": case_id},
        "problem": case["problem"],
        "config": case["config"],
        "target": target,
        "launch": launch,
        "environment": environment,
        "compiler_metadata": compiler_metadata,
        "compiled_name": getattr(compiled, "name", None),
        "stages": stage_records,
        "binary_resources": binary_inspection.get("resources", {}),
        "binary_status": binary_inspection.get("status"),
        "reports": report_records,
        "warnings": binary_inspection.get("warnings", []),
    }
    store.write_manifest(case_id, manifest)

    prepared.run()
    import torch

    torch.cuda.synchronize()
    correctness = compare_tensors(
        prepared.output,
        prepared.expected,
        atol=prepared.atol,
        rtol=prepared.rtol,
    )
    measurement: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "case_name": case["name"],
        "correctness": correctness,
        "measurements": [],
        "runtime_environment": environment,
    }
    if measure and correctness["passed"]:
        settings = case.get("measurement", {})
        warmup = int(settings.get("warmup", 10))
        samples = int(settings.get("samples", 10))
        target_sample_us = float(settings.get("target_sample_us", 5000.0))
        measurement["measurements"].append(
            benchmark_cuda_event(
                prepared.run,
                warmup=warmup,
                samples=samples,
                target_sample_us=target_sample_us,
            )
        )
        if enable_graph and bool(settings.get("cuda_graph", False)):
            try:
                measurement["measurements"].append(
                    benchmark_cuda_graph(
                        prepared.run,
                        warmup=warmup,
                        samples=samples,
                        target_sample_us=target_sample_us,
                    )
                )
            except Exception as exc:
                measurement.setdefault("warnings", []).append(
                    f"CUDA Graph measurement failed: {type(exc).__name__}: {exc}"
                )
            try:
                measurement["measurements"].append(
                    benchmark_cuda_graph_batched(
                        prepared.run,
                        warmup=warmup,
                        samples=samples,
                        target_sample_us=target_sample_us,
                        batch_size=int(settings.get("cuda_graph_batch_size", 32)),
                    )
                )
            except Exception as exc:
                measurement.setdefault("warnings", []).append(
                    f"Batched CUDA Graph measurement failed: {type(exc).__name__}: {exc}"
                )

    measurement_path = output_root / "measurements" / f"{case_id}.json"
    atomic_write_json(measurement_path, measurement)
    return {
        "case_id": case_id,
        "case_name": case["name"],
        "status": "ok" if correctness["passed"] else "correctness_failed",
        "artifact_manifest": str(store.case_dir(case_id) / "manifest.json"),
        "measurement": str(measurement_path),
        "correctness": correctness,
    }


def capture_suite(
    suite_path: str | Path,
    *,
    output_root: str | Path,
    require_cuda_major: int | None,
    cuda_bin: str | Path | None = None,
    limit: int = 0,
    measure: bool = True,
    enable_graph: bool = True,
) -> dict[str, Any]:
    suite = load_suite(suite_path)
    required_major = suite.get("requirements", {}).get("cuda_major_min")
    if require_cuda_major is None and required_major is not None:
        require_cuda_major = int(required_major)
    environment = collect_environment(require_cuda_major=require_cuda_major, cuda_bin=cuda_bin)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "environment.json", environment)

    cases = suite["cases"][:limit] if limit else suite["cases"]
    records: list[dict[str, Any]] = []
    for case in cases:
        try:
            record = capture_case(
                case,
                suite_name=suite["suite"],
                output_root=output_root,
                environment=environment,
                cuda_bin=cuda_bin,
                measure=measure,
                enable_graph=enable_graph,
            )
        except Exception as exc:
            record = {
                "case_name": case.get("name"),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        records.append(record)
        progress = f"[{len(records)}/{len(cases)}] {record.get('case_name')}: {record['status']}"
        print(progress, flush=True)

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "suite": suite["suite"],
        "suite_path": str(Path(suite_path).resolve()),
        "environment": environment,
        "records": records,
        "summary": {
            "total": len(records),
            "ok": sum(record["status"] == "ok" for record in records),
            "correctness_failed": sum(
                record["status"] == "correctness_failed" for record in records
            ),
            "error": sum(record["status"] == "error" for record in records),
        },
    }
    atomic_write_json(output_root / "run_manifest.json", run_manifest)
    return run_manifest
