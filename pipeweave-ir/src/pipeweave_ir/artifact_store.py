from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    bytes_hash,
    canonicalize_text,
)

STAGE_EXTENSIONS = {
    "ttir": ".ttir",
    "ttgir": ".ttgir",
    "llir": ".ll",
    "llvm": ".ll",
    "ptx": ".ptx",
    "cubin": ".cubin",
    "amdgcn": ".amdgcn",
    "hsaco": ".hsaco",
    "spirv": ".spv",
}


class ArtifactConflictError(RuntimeError):
    pass


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def case_dir(self, case_id: str) -> Path:
        return self.root / case_id

    def write_stages(self, case_id: str, stages: dict[str, Any]) -> dict[str, dict[str, Any]]:
        directory = self.case_dir(case_id) / "stages"
        records: dict[str, dict[str, Any]] = {}
        for name, value in sorted(stages.items()):
            extension = STAGE_EXTENSIONS.get(name, f".{name}")
            path = directory / f"kernel{extension}"
            if isinstance(value, str):
                normalized = canonicalize_text(value)
                payload = normalized.encode("utf-8")
                kind = "text"
            elif isinstance(value, bytes | bytearray | memoryview):
                payload = bytes(value)
                kind = "binary"
            else:
                continue
            digest = bytes_hash(payload)
            if path.exists():
                existing = path.read_bytes()
                if bytes_hash(existing) != digest:
                    raise ArtifactConflictError(f"existing stage differs for {case_id}: {path}")
            else:
                atomic_write_bytes(path, payload)
            records[name] = {
                "filename": str(path.relative_to(self.case_dir(case_id))),
                "kind": kind,
                "size": len(payload),
                "sha256": digest,
            }
        return records

    def write_report(self, case_id: str, name: str, text: str) -> dict[str, Any]:
        path = self.case_dir(case_id) / "reports" / name
        payload = text.encode("utf-8")
        atomic_write_text(path, text)
        return {
            "filename": str(path.relative_to(self.case_dir(case_id))),
            "size": len(payload),
            "sha256": bytes_hash(payload),
        }

    def write_manifest(self, case_id: str, manifest: dict[str, Any]) -> Path:
        path = self.case_dir(case_id) / "manifest.json"
        atomic_write_json(path, manifest)
        return path
