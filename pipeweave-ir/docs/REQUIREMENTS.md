# 下一阶段资源与数据需求

## 1. 硬件矩阵

| 优先级 | 平台 | 作用 | 最低采集量 | 状态 |
|---|---|---|---:|---|
| P0 | RTX 5090 / sm_120 | Blackwell 开发、回归与主采集 | 每 family 1,000 case | [COMPUTED] 已可用 |
| P0 | A100 / sm_80 | 第一组跨代 architecture holdout | 每 family 1,000 case | [FRAME] 待协调 |
| P1 | H100/H200 / sm_90 | Hopper、TMA/WGMMA 路线 | 每 family 1,000 case | [FRAME] 待协调 |
| P1 | RTX 4090/L40S / sm_89 | Ada、消费卡/数据中心卡差异 | 每 family 500 case | [FRAME] 待协调 |
| P2 | AMD CDNA3/4 | TTGIR+GCN 可迁移性 | Go/No-Go 后确定 | [FRAME] 暂不启动 |
| P2 | Intel Xe | TTGIR+SPIR-V 可迁移性 | Go/No-Go 后确定 | [FRAME] 暂不启动 |

[FRAME] NVIDIA 阶段至少需要 sm_80、sm_90、sm_120 三代；只有 5090 和另一张 sm_120
卡不能证明 architecture generalization。

### 每台机器的运行条件

- [FRAME] 采集期间 GPU 独占，禁止并发训练/推理任务。
- [FRAME] 保存 GPU UUID、显存容量、driver、P-state、SM/memory clock、功耗和温度。
- [FRAME] 每个 suite 至少重复三次，运行顺序随机化，并保留原始 timing samples。
- [FRAME] 5090 主标签只使用 CUDA 13.0+；当前已验证环境为 CUDA 13.2。
- [FRAME] 跨卡尽量固定 Triton/PyTorch/CUDA 版本；无法固定时将 compiler ID 作为独立变量。
- [FRAME] A100 需要确认 40 GB/80 GB、SXM/PCIe 型号，不能只记录“A100”。

## 2. Kernel 与工作负载

### 必须接入的 kernel family

1. [FRAME] 受控 Triton GEMM，用于验证 lower、资源和 cost-model 基线。
2. [FRAME] 生产 FlashAttention Triton kernel，覆盖 prefill、decode 和 causal/non-causal。
3. [FRAME] FlashInfer 中与 attention plan 对应的主 kernel、split-KV partial kernel 和 merge。
4. [FRAME] PipeWeave 当前 decomposer/scheduler 输出，用于 oracle intervention。

### Attention 维度

- [FRAME] q length：1、2–16、32–128、256–4096。
- [FRAME] KV length：短序列、1K、4K、16K、32K，以及非 tile 整数边界。
- [FRAME] head dimension：64、80、96、128、192、256。
- [FRAME] heads / KV heads：MHA、GQA、MQA，包含 40/8 等真实模型组合。
- [FRAME] batch：1、小 batch、吞吐 batch。
- [FRAME] layout：dense、ragged、paged KV。
- [FRAME] schedule：single-KV、split-KV、不同 split count、不同 CTA/tile/warps/stages。
- [FRAME] dtype：FP16、BF16；FP8 在主路径稳定后加入。

### 数据规模

- [FRAME] 第一阶段每张卡至少 2,000 个 GEMM case 和 3,000 个 attention config。
- [FRAME] 每个数学 problem 至少 4 个、推荐 8–16 个合法 config，以支持 ranking 指标。
- [FRAME] 每类边界/OOM/非法 config 也要保留状态，但不进入 latency training labels。
- [FRAME] 每个有效 case 至少 10 个原始 sample、3 次独立 run。
- [FRAME] 小于 20 µs 与不小于 20 µs 的 case 分层报告，避免 MAPE 被固定开销主导。

## 3. 标签合同

每个 case 必须明确属于以下一种 label，不允许混用：

- [FRAME] eager end-to-end launch latency；
- [FRAME] single-node CUDA Graph replay latency；
- [FRAME] batched-graph steady-state kernel-node throughput；
- [FRAME] multi-kernel aggregate latency；
- [FRAME] split-KV partial、merge、LSE/output writeback 的独立 stage latency。

[COMPUTED] 5090 首轮实验已经观察到同一小 kernel 在三种标签间相差 1–3.5 µs。
[INFERRED] 若历史训练集未保存 measurement mode，应先修复或分桶，不能直接合并重训。

## 4. Artifact 与特征

每个 compiled config 必须保存：

- [FRAME] source hash、TTIR、TTGIR、LLVM IR、PTX、cubin 与 SASS；
- [FRAME] problem、真实 grid、tile、warps、stages、num CTAs；
- [FRAME] Triton compiler metadata 中的 dynamic shared、scratch、TMEM；
- [FRAME] cuobjdump 的 static shared、register、local、stack、constant；
- [FRAME] compact category feature 与 raw opcode histogram 两套表示；
- [FRAME] kernel source ID、problem group ID、config group ID、compiler ID、architecture ID。

[COMPUTED] 5090 实验表明 static shared 与 dynamic shared 不能互相替代。

## 5. Oracle intervention 数据

为判断 FA 大 MAPE 的主要来源，每个 workload 需要生成下列配对记录：

1. [FRAME] PipeWeave 原始 decomposition + 原 scheduler + 原 feature/model；
2. [FRAME] 真实 decomposition + 其余保持不变；
3. [FRAME] 真实 schedule/grid/split count + 其余保持不变；
4. [FRAME] 真实 register/shared/occupancy feature + 其余保持不变；
5. [FRAME] 加入 merge、launch、LSE/output 和固定开销 + 其余保持不变；
6. [FRAME] 全部 oracle feature；
7. [FRAME] oracle 后剩余残差再用于数据扩充和模型重训。

每一步必须输出 ΔMAPE、ΔwMAPE、signed bias、P90 APE、Spearman、top-k recall 和 regret。

## 6. 软件与版本管理

- [FRAME] 固定 PipeWeave、Triton、FlashAttention、FlashInfer 和 PyTorch commit SHA。
- [FRAME] 保存 CUDA toolkit/ptxas/nvdisasm/cuobjdump 版本及完整 compiler ID。
- [FRAME] 生成 artifact 时不依赖扫描不透明 cache；由 CompiledKernel 直接导出。
- [FRAME] 原始 binary/IR 放对象存储或实验盘，Git 只保存代码、schema、suite generator 和报告。
- [FRAME] 使用同一 case ID 在不同 architecture/compiler 下关联，不覆盖历史记录。

## 7. 存储与执行预算

- [FRAME] 预留每张卡 20–50 GB 实验空间，用于三次重复的 IR、cubin、SASS 和 timing。
- [FRAME] 每张卡预留至少 12 小时独占采集窗口；先运行 100-case canary，再启动全量。
- [FRAME] 失败后保留 run manifest 和 traceback，禁止为提高 coverage 删除 case。
- [FRAME] 每批采集结束立即生成 checksum、coverage、correctness 和 CV 报告。

## 8. A100 到位检查单

- [ ] [FRAME] 确认 A100 型号、显存、连接方式和空闲窗口。
- [ ] [FRAME] 确认 driver 与 CUDA 13.x/PyTorch/Triton 组合可用。
- [ ] [FRAME] 同步 PipeWeave-IR commit 和固定 suite JSON。
- [ ] [FRAME] 先运行 7-case smoke，要求 artifact/correctness coverage 为 100%。
- [ ] [FRAME] 运行 100-case canary，要求 graph 标签 P90 CV 不高于 3%。
- [ ] [FRAME] 再运行全量 suite 三次，并执行 architecture-holdout。

## 9. 阶段验收

- [FRAME] 每张卡 artifact/correctness coverage 至少 95%。
- [FRAME] 大于等于 20 µs case 的 P90 跨运行 CV 不高于 3%。
- [FRAME] 所有 MAPE 同时报告 wMAPE、bias、P90 和 ranking 指标。
- [FRAME] compact TTGIR+target IR 必须在多 seed 和至少一张跨代卡上优于 shape/config。
- [FRAME] oracle intervention 能将 decomposition、schedule、resource、launch overhead 的
  ΔMAPE 分开；否则不进入大 MLP 重训。
