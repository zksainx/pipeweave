import json

from pipeweave_ir.repeatability import compare_runs


def _write_run(root, *, latency: float, ptx_hash: str) -> None:
    case_id = "case-1"
    artifact = root / "artifacts" / case_id
    measurement = root / "measurements"
    artifact.mkdir(parents=True)
    measurement.mkdir(parents=True)
    (root / "run_manifest.json").write_text(
        json.dumps({"records": [{"case_id": case_id, "case_name": "case", "status": "ok"}]})
    )
    (artifact / "manifest.json").write_text(json.dumps({"stages": {"ptx": {"sha256": ptx_hash}}}))
    (measurement / f"{case_id}.json").write_text(
        json.dumps(
            {
                "measurements": [
                    {
                        "mode": "eager_event",
                        "summary": {"median": latency, "cv": 0.01},
                    }
                ]
            }
        )
    )


def test_compare_runs_reports_hash_and_timing_stability(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, latency=10.0, ptx_hash="same")
    _write_run(second, latency=11.0, ptx_hash="same")
    result = compare_runs([first, second])
    assert result["common_case_count"] == 1
    assert result["artifact_consistency"]["ptx"]["stability_rate"] == 1.0
    timing = result["timing_repeatability"]["eager_event"]
    assert timing["comparable_cases"] == 1
    assert timing["per_case"][0]["run_medians_us"] == [10.0, 11.0]
