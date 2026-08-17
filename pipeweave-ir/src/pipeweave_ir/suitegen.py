from __future__ import annotations

from typing import Any

from .util import SCHEMA_VERSION

MEASUREMENT = {
    "warmup": 8,
    "samples": 10,
    "target_sample_us": 3000,
    "cuda_graph": True,
    "cuda_graph_batch_size": 32,
}

GEMM_PROBLEMS = (
    (32, 4096, 4096),
    (64, 2048, 4096),
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (257, 769, 513),
    (2048, 128, 4096),
    (128, 2048, 1024),
)

GEMM_CONFIGS = (
    (32, 32, 32, 4, 2),
    (64, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (128, 64, 32, 4, 3),
    (128, 128, 64, 8, 4),
)

ATTENTION_PROBLEMS = (
    (128, 64, False),
    (128, 64, True),
    (256, 64, False),
    (256, 64, True),
    (512, 64, False),
    (512, 64, True),
    (1024, 64, False),
    (1024, 64, True),
    (128, 128, True),
    (256, 128, True),
    (512, 128, True),
)

ATTENTION_CONFIGS = {
    64: (
        (32, 32, 4, 2),
        (64, 32, 4, 3),
        (64, 64, 4, 3),
        (128, 64, 8, 4),
    ),
    128: (
        (32, 32, 4, 2),
        (64, 32, 4, 3),
        (32, 64, 4, 3),
        (64, 64, 8, 3),
    ),
}


def build_rtx5090_day6_suite() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for m, n, k in GEMM_PROBLEMS:
        for block_m, block_n, block_k, num_warps, num_stages in GEMM_CONFIGS:
            cases.append(
                {
                    "name": (
                        f"gemm_m{m}_n{n}_k{k}_bm{block_m}_bn{block_n}_bk{block_k}"
                        f"_w{num_warps}_s{num_stages}"
                    ),
                    "kernel": "matmul",
                    "family": "gemm",
                    "problem": {"m": m, "n": n, "k": k, "dtype": "float16"},
                    "config": {
                        "block_m": block_m,
                        "block_n": block_n,
                        "block_k": block_k,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    },
                    "measurement": dict(MEASUREMENT),
                }
            )

    for n_ctx, head_dim, causal in ATTENTION_PROBLEMS:
        for block_m, block_n, num_warps, num_stages in ATTENTION_CONFIGS[head_dim]:
            causal_name = "causal" if causal else "full"
            cases.append(
                {
                    "name": (
                        f"attention_n{n_ctx}_d{head_dim}_{causal_name}_bm{block_m}_bn{block_n}"
                        f"_w{num_warps}_s{num_stages}"
                    ),
                    "kernel": "attention",
                    "family": "flash_attention_like",
                    "problem": {
                        "batch": 1,
                        "heads": 8,
                        "n_ctx": n_ctx,
                        "head_dim": head_dim,
                        "dtype": "float16",
                        "causal": causal,
                    },
                    "config": {
                        "block_m": block_m,
                        "block_n": block_n,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    },
                    "measurement": dict(MEASUREMENT),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": "rtx5090_day6",
        "requirements": {
            "vendor": "nvidia",
            "cuda_major_min": 13,
            "compute_capability": "12.0",
        },
        "cases": cases,
    }
