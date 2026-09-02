from pipeweave_ir.nvidia import parse_cuobjdump_resource_usage


def test_parse_cuobjdump_resource_usage() -> None:
    report = """
Resource usage:
 Common:
  GLOBAL:0
 Function _kernel_a:
  REG:32 STACK:0 SHARED:2048 LOCAL:16 CONSTANT[0]:512
 Function _kernel_b:
  REG:64 STACK:8 SHARED:1024 LOCAL:0 CONSTANT[0]:256
"""
    parsed = parse_cuobjdump_resource_usage(report)
    assert parsed["functions"]["_kernel_a"]["reg"] == 32
    assert parsed["functions"]["_kernel_b"]["stack"] == 8
    assert parsed["maxima"]["reg"] == 64
    assert parsed["maxima"]["shared"] == 2048
