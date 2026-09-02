from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only parsers and tests do not require Triton.
    triton = None
    tl = None


matmul_kernel = None
attention_kernel = None


if triton is not None:

    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m

        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offsets_m[:, None] * K + offsets_k[None, :]
        b_ptrs = b_ptr + offsets_k[:, None] * N + offsets_n[None, :]

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            k_offsets = k_block * BLOCK_K + offsets_k
            a = tl.load(
                a_ptrs,
                mask=(offsets_m[:, None] < M) & (k_offsets[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(k_offsets[:, None] < K) & (offsets_n[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N

        c_ptrs = c_ptr + offsets_m[:, None] * N + offsets_n[None, :]
        c_mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    @triton.jit
    def attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        softmax_scale,
        BH: tl.constexpr,
        N_CTX: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(axis=0)
        bh_index = tl.program_id(axis=1)
        offsets_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tl.arange(0, BLOCK_N)
        offsets_d = tl.arange(0, HEAD_DIM)
        base = bh_index * N_CTX * HEAD_DIM

        q_ptrs = q_ptr + base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]
        q = tl.load(q_ptrs, mask=offsets_m[:, None] < N_CTX, other=0.0)
        q = (q * (softmax_scale * 1.4426950408889634)).to(tl.float16)

        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for start_n in range(0, N_CTX, BLOCK_N):
            current_n = start_n + offsets_n
            k_ptrs = k_ptr + base + current_n[:, None] * HEAD_DIM + offsets_d[None, :]
            v_ptrs = v_ptr + base + current_n[:, None] * HEAD_DIM + offsets_d[None, :]
            valid_k = current_n[:, None] < N_CTX
            k = tl.load(k_ptrs, mask=valid_k, other=0.0)
            v = tl.load(v_ptrs, mask=valid_k, other=0.0)

            scores = tl.dot(q, tl.trans(k))
            valid_scores = current_n[None, :] < N_CTX
            if CAUSAL:
                valid_scores = valid_scores & (current_n[None, :] <= offsets_m[:, None])
            scores = tl.where(valid_scores, scores, -float("inf"))

            next_max = tl.maximum(row_max, tl.max(scores, axis=1))
            probabilities = tl.exp2(scores - next_max[:, None])
            alpha = tl.exp2(row_max - next_max)
            accumulator = accumulator * alpha[:, None]
            accumulator += tl.dot(probabilities.to(tl.float16), v)
            row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
            row_max = next_max

        output = accumulator / row_sum[:, None]
        output_ptrs = o_ptr + base + offsets_m[:, None] * HEAD_DIM + offsets_d[None, :]
        tl.store(output_ptrs, output, mask=offsets_m[:, None] < N_CTX)


def require_triton() -> None:
    if triton is None or matmul_kernel is None or attention_kernel is None:
        raise RuntimeError("Triton is required for GPU kernel capture")
