from pipeweave_ir.suitegen import build_rtx5090_day6_suite


def test_day6_suite_has_multiple_configs_per_problem() -> None:
    suite = build_rtx5090_day6_suite()
    names = [case["name"] for case in suite["cases"]]
    assert len(names) == 89
    assert len(names) == len(set(names))

    gemm = [case for case in suite["cases"] if case["family"] == "gemm"]
    attention = [case for case in suite["cases"] if case["family"] == "flash_attention_like"]
    assert len(gemm) == 45
    assert len(attention) == 44
