import pytest

from pipeweave_ir.benchmark import summarize_samples
from pipeweave_ir.metrics import latency_metrics, ranking_metrics


def test_latency_metrics() -> None:
    metrics = latency_metrics([10.0, 20.0], [11.0, 18.0])
    assert metrics["mape"] == pytest.approx(10.0)
    assert metrics["wmape"] == pytest.approx(10.0)
    assert metrics["signed_bias"] == pytest.approx(0.0)


def test_ranking_metrics() -> None:
    metrics = ranking_metrics([10.0, 12.0, 20.0], [11.0, 10.0, 21.0], top_k=2)
    assert metrics["top_k_recall"] == 1.0
    assert metrics["regret_percent"] == pytest.approx(20.0)


def test_sample_summary_preserves_noise_statistics() -> None:
    summary = summarize_samples([10.0, 10.0, 11.0])
    assert summary["median"] == 10.0
    assert summary["max"] == 11.0
    assert summary["cv"] > 0.0
