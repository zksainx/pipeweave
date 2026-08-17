import json
from pathlib import Path

import pytest

from pipeweave_ir.capture import load_suite


def test_smoke_suite_is_valid() -> None:
    root = Path(__file__).parents[1]
    suite = load_suite(root / "configs" / "rtx5090_smoke.json")
    assert suite["suite"] == "rtx5090_smoke"
    assert {case["kernel"] for case in suite["cases"]} == {"matmul", "attention"}


def test_duplicate_case_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cases": [{"name": "same"}, {"name": "same"}],
            }
        )
    )
    with pytest.raises(ValueError, match="unique"):
        load_suite(path)
