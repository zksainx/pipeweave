from __future__ import annotations

import glob
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from .util import SCHEMA_VERSION, parse_major_version, utc_now


def run_command(command: list[str], *, timeout: int = 20) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": command,
        "status": "ok" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _cuda_candidate_dirs() -> list[Path]:
    candidates: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if value := os.environ.get(variable):
            candidates.append(Path(value) / "bin")
    candidates.append(Path("/usr/local/cuda/bin"))
    versioned = sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    candidates.extend(Path(item) / "bin" for item in versioned)
    return candidates


def find_cuda_tool(name: str, cuda_bin: str | Path | None = None) -> str | None:
    if cuda_bin:
        candidate = Path(cuda_bin) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    if path := shutil.which(name):
        return path
    for directory in _cuda_candidate_dirs():
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def collect_environment(
    *, require_cuda_major: int | None = None, cuda_bin: str | Path | None = None
) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "tools": {},
        "warnings": [],
    }

    try:
        import torch

        environment["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        cuda_major = parse_major_version(torch.version.cuda)
        if require_cuda_major is not None and (
            cuda_major is None or cuda_major < require_cuda_major
        ):
            raise RuntimeError(
                f"CUDA runtime {torch.version.cuda!r} does not satisfy required major "
                f">= {require_cuda_major}"
            )
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            environment["gpu"] = {
                "index": index,
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "multiprocessor_count": properties.multi_processor_count,
                "total_memory_bytes": properties.total_memory,
            }
    except ImportError as exc:
        environment["torch"] = {"status": "missing", "error": str(exc)}
        if require_cuda_major is not None:
            raise RuntimeError("PyTorch is required to enforce the CUDA runtime version") from exc

    try:
        import triton

        environment["triton"] = {"version": triton.__version__}
        try:
            target = triton.runtime.driver.active.get_current_target()
            environment["triton"]["target"] = {
                "backend": target.backend,
                "arch": target.arch,
                "warp_size": target.warp_size,
            }
        except Exception as exc:  # Driver initialization failures belong in the manifest.
            warning = f"Triton target unavailable: {type(exc).__name__}: {exc}"
            environment["warnings"].append(warning)
    except ImportError as exc:
        environment["triton"] = {"status": "missing", "error": str(exc)}

    for tool in ("nvcc", "ptxas", "cuobjdump", "nvdisasm"):
        path = find_cuda_tool(tool, cuda_bin)
        if path is None:
            environment["tools"][tool] = {"status": "missing"}
            continue
        version_flag = "--version"
        result = run_command([path, version_flag])
        environment["tools"][tool] = {"path": path, "version": result}

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        environment["nvidia_smi"] = run_command(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,name,driver_version,pstate,clocks.current.sm,"
                "clocks.current.memory,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    else:
        environment["warnings"].append("nvidia-smi not found")

    return environment
