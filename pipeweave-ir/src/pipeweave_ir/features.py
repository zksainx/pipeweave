from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MLIR_OPERATION = re.compile(
    r'"?((?:tt|ttg|ttng|arith|math|scf|cf|gpu|nvgpu|nvvm|llvm)\.[A-Za-z0-9_.]+)"?'
)
MLIR_LAYOUT = re.compile(r"#[A-Za-z_][A-Za-z0-9_]*")
PTX_INSTRUCTION = re.compile(r"^(?:@[!%A-Za-z0-9_$.]+\s+)?([A-Za-z][A-Za-z0-9_.]+)\b")
SASS_INSTRUCTION = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(?:@[!A-Za-z0-9_]+\s+)?([A-Z][A-Z0-9_.]+)\b")


def extract_mlir_features(text: str) -> dict[str, Any]:
    operations: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for line in text.splitlines():
        operations.update(MLIR_OPERATION.findall(line))
        layouts.update(MLIR_LAYOUT.findall(line))

    categories = {
        "dot": sum(value for key, value in operations.items() if "dot" in key or "mma" in key),
        "load": sum(value for key, value in operations.items() if "load" in key),
        "store": sum(value for key, value in operations.items() if "store" in key),
        "layout_convert": sum(
            value for key, value in operations.items() if "convert_layout" in key
        ),
        "barrier": sum(value for key, value in operations.items() if "barrier" in key),
        "async": sum(
            value
            for key, value in operations.items()
            if any(token in key for token in ("async", "tma", "mbarrier"))
        ),
        "loop": sum(value for key, value in operations.items() if key.startswith("scf.for")),
    }
    return {
        "line_count": len(text.splitlines()),
        "operation_count": sum(operations.values()),
        "operations": dict(sorted(operations.items())),
        "layouts": dict(sorted(layouts.items())),
        "categories": categories,
    }


def _ptx_category(instruction: str) -> str:
    base = instruction.lower()
    if any(token in base for token in ("mma", "wgmma", "tcgen")):
        return "tensor"
    if base.startswith(("ld", "mov")):
        return "load"
    if base.startswith("st"):
        return "store"
    if base.startswith(("atom", "red")):
        return "atomic"
    if any(token in base for token in ("barrier", "mbarrier")):
        return "barrier"
    if any(token in base for token in ("cp.async", "tma", "async")):
        return "async"
    if base.startswith(("bra", "brx", "call", "ret", "exit")):
        return "control"
    if any(token in base for token in ("ex2", "lg2", "rcp", "rsqrt", "sqrt", "sin", "cos")):
        return "special"
    if base.startswith(("add", "sub", "mul", "mad", "fma", "max", "min", "set", "selp")):
        return "alu"
    return "other"


def extract_ptx_features(text: str) -> dict[str, Any]:
    instructions: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith(".") or line.endswith(":") or line in {"{", "}"}:
            continue
        match = PTX_INSTRUCTION.match(line)
        if not match:
            continue
        instruction = match.group(1)
        instructions[instruction] += 1
        categories[_ptx_category(instruction)] += 1
    return {
        "line_count": len(text.splitlines()),
        "instruction_count": sum(instructions.values()),
        "instructions": dict(sorted(instructions.items())),
        "categories": dict(sorted(categories.items())),
    }


def _sass_category(instruction: str) -> str:
    base = instruction.upper()
    if any(token in base for token in ("HMMA", "IMMA", "WGMMA", "TCGEN", "MMA")):
        return "tensor"
    if base.startswith(("LDGSTS", "CPASYNC", "TMA")):
        return "async_copy"
    if base.startswith(("LDG", "ULDG")):
        return "global_load"
    if base.startswith(("STG", "USTG")):
        return "global_store"
    if base.startswith(("LDS", "ULDS")):
        return "shared_load"
    if base.startswith(("STS", "USTS")):
        return "shared_store"
    if base.startswith(("LDL", "STL", "LDC", "ULDC")):
        return "local_or_constant"
    if base.startswith(("BAR", "DEPBAR", "WARPSYNC")):
        return "barrier"
    if base.startswith("MUFU"):
        return "special"
    if base.startswith(("BRA", "BRX", "BSSY", "BSYNC", "CALL", "RET", "EXIT")):
        return "control"
    if base.startswith(
        ("IADD", "IMAD", "FADD", "FMUL", "FFMA", "F2F", "LOP", "SHF", "LEA", "ISETP")
    ):
        return "alu"
    return "other"


def extract_sass_features(text: str) -> dict[str, Any]:
    instructions: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    for instruction in SASS_INSTRUCTION.findall(text):
        instructions[instruction] += 1
        categories[_sass_category(instruction)] += 1
    return {
        "line_count": len(text.splitlines()),
        "instruction_count": sum(instructions.values()),
        "instructions": dict(sorted(instructions.items())),
        "categories": dict(sorted(categories.items())),
    }


def flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, int | float):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flatten_numeric(child, item, output)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            flatten_numeric(child, item, output)


def extract_artifact_features(case_dir: str | Path) -> dict[str, Any]:
    case_dir = Path(case_dir)
    manifest = json.loads((case_dir / "manifest.json").read_text())
    stages = manifest.get("stages", {})
    record: dict[str, Any] = {
        "schema_version": manifest.get("schema_version"),
        "case_id": manifest["identity"]["case_id"],
        "kernel": manifest["identity"].get("kernel"),
        "family": manifest["identity"].get("family"),
        "problem": manifest.get("problem", {}),
        "config": manifest.get("config", {}),
        "features": {},
    }

    for stage_name in ("ttir", "ttgir", "llir"):
        if stage_name not in stages:
            continue
        path = case_dir / stages[stage_name]["filename"]
        record["features"][stage_name] = extract_mlir_features(path.read_text(errors="replace"))
    if "ptx" in stages:
        path = case_dir / stages["ptx"]["filename"]
        record["features"]["ptx"] = extract_ptx_features(path.read_text(errors="replace"))
    nvdisasm = manifest.get("reports", {}).get("nvdisasm")
    if nvdisasm and nvdisasm.get("filename"):
        path = case_dir / nvdisasm["filename"]
        record["features"]["sass"] = extract_sass_features(path.read_text(errors="replace"))
    record["features"]["compiler"] = manifest.get("compiler_metadata", {})
    record["features"]["launch"] = manifest.get("launch", {})
    record["features"]["binary"] = manifest.get("binary_resources", {})
    record["features"]["artifact_sizes"] = {
        name: stage.get("size", 0) for name, stage in stages.items()
    }

    flat: dict[str, float] = {}
    flatten_numeric("", record["features"], flat)
    record["flat_features"] = flat
    return record
