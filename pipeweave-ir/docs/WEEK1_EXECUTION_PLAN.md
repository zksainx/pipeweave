# 第一周执行计划：NVIDIA/RTX 5090 多层 IR MVP

> [FRAME] 周期：5 个工作日。
> [FRAME] 主要平台：RTX 5090，CUDA 13.2，Triton 3.7.1。
> [FRAME] 第一周目标：打通“case → compile artifacts → binary resources → correctness → timing → features → report”的完整链路。

## 第一周必须回答的问题

- [FRAME] Q1：能否对同一个 Triton config 稳定保存 TTIR、TTGIR、LLVM IR、PTX 和 cubin？
- [FRAME] Q2：能否从最终 cubin 恢复寄存器、shared/local memory 等资源信息？
- [FRAME] Q3：相同输入重复编译后，canonical text/hash 是否稳定？
- [FRAME] Q4：TTGIR 与 PTX 是否分别包含独立、可解释的 schedule/lowering 信息？
- [FRAME] Q5：计时协议能否区分 eager、CUDA Graph、单 kernel 与多 kernel？

## Day 1：范围冻结、schema 与可复现环境

### 任务

- [x] [FRAME] 创建独立目录 `pipeweave-ir/`，不修改现有 PipeWeave 模型代码。
- [x] [FRAME] 固定 RTX 5090 环境：Python、PyTorch、CUDA、Triton、driver、GPU UUID、clock/power 状态。
- [x] [FRAME] 定义 suite、case、artifact manifest 和 measurement record schema。
- [x] [FRAME] 建立 CPU-only 单元测试，确保无 GPU 环境也能验证 parser、hash 和 metrics。
- [x] [FRAME] 选择第一批 case：至少 4 个 GEMM 和 3 个 Flash-Attention-like config。

### 交付物

- [x] `pyproject.toml`
- [x] `configs/rtx5090_smoke.json`
- [x] `docs/DATA_SCHEMA.md`
- [x] `src/pipeweave_ir/environment.py`
- [x] manifest/schema/parser 单元测试

### 验收

- [x] [FRAME] `python -m pipeweave_ir env --require-cuda-major 13` 在 5090 成功。
- [x] [FRAME] 环境 JSON 包含 PyTorch CUDA 版本、Triton 版本、GPU 名称、compute capability 和工具路径。
- [x] [FRAME] 本地 CPU-only tests 全部通过。

## Day 2：Triton 编译链与 artifact store

### 任务

- [x] [FRAME] 使用 `JITFunction.run(..., warmup=True)` 获取 `CompiledKernel`，不依赖执行后扫描 cache。
- [x] [FRAME] 保存 `compiled.asm` 暴露的所有 stage，而不是硬编码只有 PTX/cubin。
- [x] [FRAME] 对文本 IR 规范化换行、临时路径和尾部空白，并计算 SHA-256。
- [x] [FRAME] 以稳定 `case_id` 建立 content-addressed artifact 目录。
- [x] [FRAME] 保存 compiler metadata、source hash、shape、config、grid 和 target。

### 交付物

- [x] `artifact_store.py`
- [x] `capture.py`
- [x] 一个 GEMM case 的完整 artifact manifest

### 验收

- [x] [FRAME] 同一 case 重跑不会覆盖成不一致内容。
- [x] [FRAME] manifest 中每个 stage 都有 filename、size 和 SHA-256。
- [x] [FRAME] capture 失败会写入明确 error record，不静默跳过。

## Day 3：NVIDIA 最终二进制资源与静态特征

### 任务

- [x] [FRAME] 自动定位 CUDA 13.x 的 `cuobjdump` 和 `nvdisasm`。
- [x] [FRAME] 保存原始 resource/disassembly report。
- [x] [FRAME] 解析 function-level registers、stack、shared、local 和 constant memory。
- [x] [FRAME] 提取 TTGIR op histogram、layout/memory-space 和 synchronization 特征。
- [x] [FRAME] 提取 PTX tensor/load/store/atomic/barrier/async/control/special-function 特征。

### 交付物

- [x] `nvidia.py`
- [x] `features.py`
- [x] parser unit tests
- [x] `features.jsonl`

### 验收

- [x] [FRAME] parser 对真实 CUDA 13.2 输出和测试 fixture 都可工作。
- [x] [FRAME] 一个 case 能输出 TTGIR、PTX、binary 三个 feature namespace。
- [x] [FRAME] 找不到可选工具时记录 warning，但保留其余 artifact。

## Day 4：正确性与计时协议

### 任务

- [x] [FRAME] GEMM 与 PyTorch reference 对比 correctness。
- [x] [FRAME] Flash-Attention-like kernel 与 PyTorch FP32 reference 对比 correctness。
- [x] [FRAME] 保存 eager event 原始样本、median、P10、P90、MAD 和 CV。
- [x] [FRAME] 可选采集 CUDA Graph replay，使用独立 label。
- [x] [FRAME] 预分配输入、输出和 scratch，禁止 allocation 进入计时区间。
- [x] [FRAME] 记录 grid、block threads、shared memory 和每次 launch 的 config。

### 交付物

- [x] `benchmark.py`
- [x] 第一批 5090 measurement records
- [x] correctness summary

### 验收

- [x] [FRAME] correctness 失败的 case 不写入可训练 measurement 集合。
- [x] [FRAME] eager 与 graph 结果不共用同一个 duration 字段。
- [x] [FRAME] 所有统计值可以由 raw samples 重新计算。

## Day 5：首轮运行、误差观察与周报

### 任务

- [x] [FRAME] 在 5090 运行完整 smoke suite。
- [x] [FRAME] 按 kernel family、shape 和 config 汇总 capture/correctness/measurement 成功率。
- [x] [FRAME] 对比不同 config 的 TTGIR、PTX 和 binary resource 差异。
- [x] [FRAME] 生成首轮 static-feature 与 latency 表。
- [x] [FRAME] 记录所有失败 case 及其根因，不为提高成功率删除失败项。
- [x] [FRAME] 形成 Week 1 report 和 Week 2 调整项。

### 交付物

- [x] `reports/RTX5090_WEEK1.md`
- [x] smoke suite 的 run manifest、index 和 feature table
- [x] 更新后的 Day 6–10 backlog

### 第一周 Go/No-Go

- [x] [FRAME] 至少 90% smoke cases 完成 compile artifact 捕获。
- [x] [FRAME] 至少 90% smoke cases correctness 通过并得到稳定 timing。
- [x] [FRAME] 至少能展示 3 个 config 在 TTGIR 与 PTX/binary 上的可解释差异。
- [x] [FRAME] 若 TTGIR 或 cubin 捕获不稳定，第二周优先修 capture，不扩展数据规模。
- [x] [FRAME] 若绝对 latency 噪声过大但 config ranking 稳定，第二周同时保留 ranking 路线。

## 每日评审模板

每日结束时回答：

1. [COMPUTED] 今天新增多少成功/失败 case？
2. [COMPUTED] 哪个 stage 的 capture coverage 最低？
3. [COMPUTED] correctness 和 timing 的噪声是否满足门槛？
4. [INFERRED] 当前最大不确定性属于 compiler、hardware、measurement 还是 model？
5. [FRAME] 明天只优先解决哪个不确定性？
