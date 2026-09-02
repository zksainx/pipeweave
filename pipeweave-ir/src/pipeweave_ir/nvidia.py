from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from .environment import find_cuda_tool

RESOURCE_TOKEN = re.compile(r"\b(REG|STACK|SHARED|LOCAL|CONSTANT\[\d+\]):\s*(\d+)")
FUNCTION_HEADER = re.compile(r"^\s*Function\s+([^:]+):\s*$")


def parse_cuobjdump_resource_usage(text: str) -> dict[str, Any]:
    functions: dict[str, dict[str, int]] = {}
    current = "<common>"
    functions[current] = {}
    for line in text.splitlines():
        if match := FUNCTION_HEADER.match(line):
            current = match.group(1).strip()
            functions.setdefault(current, {})
            continue
        for key, value in RESOURCE_TOKEN.findall(line):
            functions[current][key.lower().replace("[", "_").replace("]", "")] = int(value)

    functions = {name: values for name, values in functions.items() if values}
    maxima: dict[str, int] = {}
    for values in functions.values():
        for key, value in values.items():
            maxima[key] = max(maxima.get(key, 0), value)
    return {"functions": functions, "maxima": maxima}


def _run_tool(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "error", "command": command, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "status": "ok" if result.returncode == 0 else "error",
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def inspect_cubin(cubin: str | Path, *, cuda_bin: str | Path | None = None) -> dict[str, Any]:
    cubin = Path(cubin)
    result: dict[str, Any] = {"status": "ok", "warnings": [], "reports": {}}

    cuobjdump = find_cuda_tool("cuobjdump", cuda_bin)
    if cuobjdump:
        report = _run_tool([cuobjdump, "--dump-resource-usage", str(cubin)])
        result["reports"]["cuobjdump_resource"] = report
        if report["status"] == "ok":
            combined = f"{report.get('stdout', '')}\n{report.get('stderr', '')}"
            result["resources"] = parse_cuobjdump_resource_usage(combined)
        else:
            result["warnings"].append("cuobjdump resource extraction failed")
    else:
        result["warnings"].append("cuobjdump not found")

    nvdisasm = find_cuda_tool("nvdisasm", cuda_bin)
    if nvdisasm:
        report = _run_tool([nvdisasm, "-g", "-c", str(cubin)])
        result["reports"]["nvdisasm"] = report
        if report["status"] != "ok":
            result["warnings"].append("nvdisasm failed")
    else:
        result["warnings"].append("nvdisasm not found")

    if result["warnings"]:
        result["status"] = "partial"
    return result
