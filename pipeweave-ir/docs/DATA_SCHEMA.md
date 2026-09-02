# PipeWeave-IR Artifact 与 Measurement Schema

## 设计原则

- [FRAME] 编译 artifact 与运行 measurement 分开保存，通过稳定 `case_id` 关联。
- [FRAME] 原始 IR/binary 使用独立文件；manifest 只保存路径、hash、size 和 provenance。
- [FRAME] 原始 timing samples 必须保留，汇总统计不能取代原始数据。
- [FRAME] eager、graph、kernel-stage 和 aggregate 采用不同 measurement record。
- [FRAME] case ID 不包含时间戳，因此同一 source/shape/config/target 可稳定复现。

## Suite

```json
{
  "schema_version": "1.0",
  "suite": "rtx5090_smoke",
  "requirements": {
    "vendor": "nvidia",
    "cuda_major_min": 13,
    "compute_capability": "12.0"
  },
  "cases": []
}
```

## Case

```json
{
  "name": "gemm_256x256x256_bm64_bn64",
  "kernel": "matmul",
  "family": "gemm",
  "problem": {
    "m": 256,
    "n": 256,
    "k": 256,
    "dtype": "float16"
  },
  "config": {
    "block_m": 64,
    "block_n": 64,
    "block_k": 32,
    "num_warps": 4,
    "num_stages": 3
  },
  "measurement": {
    "warmup": 10,
    "samples": 15,
    "target_sample_us": 5000,
    "cuda_graph": true
  }
}
```

## Artifact manifest

必需 namespace：

- [FRAME] `identity`：case ID、suite、kernel/family、source hash；
- [FRAME] `environment`：host、GPU、driver、CUDA、PyTorch、Triton、tool paths；
- [FRAME] `target`：vendor、architecture、compute capability；
- [FRAME] `problem` 与 `config`；
- [FRAME] `launch`：grid、num_warps、threads、num_stages；
- [FRAME] `compiler_metadata`；
- [FRAME] `stages`：每个 IR/binary 的 filename、hash、size、kind；
- [FRAME] `binary_resources`：解析结果和原始报告文件；
- [FRAME] `status`、`warnings` 和 error traceback。

## Measurement record

```json
{
  "schema_version": "1.0",
  "case_id": "sha256-prefix",
  "mode": "eager_event",
  "unit": "us",
  "raw_samples": [12.1, 12.0, 12.2],
  "summary": {
    "median": 12.1,
    "p10": 12.0,
    "p90": 12.2,
    "mad": 0.1,
    "cv": 0.008
  },
  "correctness": {
    "passed": true,
    "max_abs_error": 0.01,
    "max_rel_error": 0.02
  },
  "runtime_environment": {}
}
```

允许的 `mode` 初始集合：

- `eager_event`
- `cuda_graph_event`
- `cuda_graph_batched_event`（同一 graph 内包含多个 kernel node，报告每 node steady-state throughput）
- `profiler_kernel_stage`
- `kernel_sum`
- `multi_kernel_aggregate`

## Feature record

- [FRAME] `ttgir.*`：op histogram、dot/load/store/layout/convert/barrier/async、loop/region 数；
- [FRAME] `target_ir.*`：PTX instruction histogram 与 tensor/memory/control 类别；
- [FRAME] `binary.*`：register/shared/local/stack/constant、binary size；
- [FRAME] `hardware.*`：SM、clock、bandwidth、cache、tensor/vector throughput；
- [FRAME] `label.*`：通过 case ID 和 measurement mode 后连接，不存入 compile-only artifact。

## 划分字段

- [FRAME] `problem_group_id`：同一数学问题的所有 config；
- [FRAME] `kernel_source_id`：同一 source hash；
- [FRAME] `compiler_id`：Triton/backend/toolchain 组合；
- [FRAME] `architecture_id`：例如 `nvidia-sm120`；
- [FRAME] 后续训练必须按这些 group 划分，而不是随机逐行切分。
