# RTX 5090 Week 1 实验报告

> [COMPUTED] 日期：2026-08-17。
> [COMPUTED] 平台：NVIDIA GeForce RTX 5090（sm_120）、driver 595.71.05、PyTorch
> 2.13.0+cu132、CUDA 13.2、Triton 3.7.1。
> [FRAME] 本报告验证的是 PipeWeave-IR 捕获链和自带的 Flash-Attention-like Triton
> kernel；它不是生产版 FlashAttention/FlashInfer 的最终误差报告。

## 结论先行

1. [COMPUTED] 多层 IR 捕获链已经跑通：89/89 case 完成 TTIR、TTGIR、LLVM IR、
   PTX、cubin、SASS、资源、正确性和 timing 采集；三次独立运行中所有 stage hash
   的稳定率均为 100%。
2. [COMPUTED] 小 kernel 的标签语义会显著改变观测值。以 GEMM 128×128×128
   为例，同一 kernel 的 eager、单节点 CUDA Graph、32 节点 batched graph 分别为
   5.585、4.097 和 2.047 µs。
3. [COMPUTED] 三次 89-case 重复实验中，single-graph 的跨运行 CV 最大为 2.35%，
   而 eager 有 13/89 case 超过 3%，最大达到 24.51%。
4. [COMPUTED] 在 attention 的初步 shape-holdout 上，紧凑 TTGIR+PTX 特征对 batched
   graph 标签的 MAPE 为 14.76%，低于含真实 grid 的 shape/config baseline 15.93%；
   386 维原始 combined 特征反而达到 21.74%。
5. [INFERRED] 当前首要问题不是“是否要收集更多 opcode”，而是先固定标签合同、保留
   动态资源、压缩高维共线特征，并把生产版 FA 的真实 decomposition/launch 序列接入。

## 数据覆盖

| Suite | Case | 运行次数 | 编译成功 | 正确性通过 | 标签模式 |
|---|---:|---:|---:|---:|---|
| smoke | 7 | 3 | 21/21 | 21/21 | eager、single graph、batched graph |
| day6 matrix | 89 | 3 | 267/267 | 267/267 | eager、single graph |
| day6 batched | 89 | 1 | 89/89 | 89/89 | eager、single graph、batched graph |

[COMPUTED] Day 6 矩阵包含 45 个 GEMM case 和 44 个 attention case；同一数学问题
有 4–5 个 tile/warp/stage config，因此可以做 problem-grouped split 和 config ranking。

## Artifact 与资源完整性

[COMPUTED] 在 89 个共同 case 的三次运行中，下列 artifact hash 全部一致：

| Stage | 可比较 case | hash 稳定率 |
|---|---:|---:|
| TTIR | 89 | 100% |
| TTGIR | 89 | 100% |
| LLVM IR | 89 | 100% |
| PTX | 89 | 100% |
| cubin | 89 | 100% |
| source | 89 | 100% |

[COMPUTED] 首版实现曾把 Triton KernelMetadata 当成普通 tuple 序列，字段名因此丢失。
修正 namedtuple 归一化顺序后，首个 GEMM 的 dynamic_shared_bytes 被正确恢复为
16 KiB；cuobjdump 单独报告的 cubin 静态 shared 段只有 1 KiB。

[INFERRED] 若只使用 cubin 静态 SHARED，occupancy/active-CTA 特征会系统性错误；
compiler launch metadata 与最终 binary resource 必须同时保留，不能互相替代。

同一个 Attention N=256、D=64、causal problem 的 lowering 差异如下：

| BM×BN / warps / stages | dynamic SMEM | registers | TTGIR ops | PTX inst. | SASS inst. | batched µs |
|---|---:|---:|---:|---:|---:|---:|
| 32×32 / 4 / 2 | 14 KiB | 95 | 275 | 516 | 528 | 6.337 |
| 64×32 / 4 / 3 | 24 KiB | 121 | 327 | 680 | 704 | 5.325 |
| 64×64 / 4 / 3 | 40 KiB | 156 | 319 | 903 | 912 | **5.184** |
| 128×64 / 8 / 4 | 64 KiB | 162 | 366 | 878 | 904 | 8.324 |

[COMPUTED] 最大 tile 并非最快：128×64 config 使用最高动态 shared/register，最终比
64×64 config 慢 60.6%。[INFERRED] 因而只靠 FLOPs/bytes 或上层 tile 名称不足以恢复
实际 strategy；TTGIR lowering、PTX/SASS 指令和资源约束需要共同参与。

## 标签语义与重复性

### 三种标签

- [KNOWN] eager_event：CUDA event 内由 host 逐次提交 Triton launch。
- [KNOWN] cuda_graph_event：每个 graph 含一个 kernel node，逐次 replay。
- [KNOWN] cuda_graph_batched_event：每个 graph 含 32 个同 stream kernel node，报告
  每 node 的 steady-state throughput；它不是端到端 eager latency。

代表性 smoke 结果如下：

| Case | eager (µs) | single graph (µs) | batched graph/node (µs) |
|---|---:|---:|---:|
| GEMM 128³ | 5.585 | 4.097 | 2.047 |
| Attention N=128, D=64 | 6.027 | 4.099 | 3.714 |
| Attention N=256, D=64 | 6.007 | 6.146 | 5.182 |
| Attention N=512, D=64 | 14.841 | 16.381 | 14.728 |

### 跨运行稳定性

| 数据集/模式 | median CV | P90 CV | max CV | CV > 3% |
|---|---:|---:|---:|---:|
| 89-case eager | 0.321% | 15.694% | 24.511% | 13/89 |
| 89-case single graph | 0.022% | 0.104% | 2.353% | 0/89 |
| 7-case batched graph | 0.015% | 0.098% | 0.208% | 0/7 |

[COMPUTED] eager 的高波动 case 常在约 6 µs 与约 9–9.6 µs 两个稳定带之间跳变，
但各次运行内部 CV 往往很低。

[INFERRED] 这组离散跳变与“同一运行内 kernel 算术吞吐随机变化”不符，更符合 host
submission、driver doorbell 或 launch bubble 被 CUDA event 时间线吸收的现象。对 5–10 µs
kernel，约 3 µs 的标签差异本身就可产生 30%–60% 的相对误差。

## Attention 初步消融

[FRAME] 以下使用 44 个 attention case、按数学问题做 shape-holdout；训练集 32 行，
测试集 12 行，固定 seed=17、ridge alpha=1。该实验用于发现方向，不用于论文显著性结论。

| Feature set | retained features | eager MAPE | single graph MAPE | batched graph MAPE |
|---|---:|---:|---:|---:|
| shape/config | 12 | 16.60% | 15.87% | 15.93% |
| compact TTGIR | 18 | 16.95% | 15.17% | 15.53% |
| compact PTX | 22 | 16.83% | 15.70% | 15.29% |
| compact TTGIR+PTX | 28 | 16.66% | 15.21% | 14.76% |
| compact TTGIR+PTX+SASS/resources | 52 | **14.64%** | **14.94%** | **14.42%** |
| raw combined | 386 | 27.08% | 23.24% | 21.74% |

[COMPUTED] 对 batched graph 标签，compact TTGIR+PTX 相对 shape/config baseline 降低
1.17 个 MAPE 百分点；测试中三个 problem group 的平均 Spearman 为 0.933、best-config
regret 为 0%。

[INFERRED] 该结果支持“TTGIR 与 PTX 可以提供互补信号”的可行性，但当前 test 只有三个
problem group，ranking 恰好全对不能外推。下一轮必须报告多 seed、更多 shape 和 A100
architecture holdout。

[COMPUTED] raw combined 有 386 个非恒定特征，而训练行只有 32；它明显劣于 28–52 维的
compact 版本。[INFERRED] 这说明当前失败主要是样本量与特征维数/共线性不匹配，而不是
“组合多个 IR 层必然无效”。

## 对 Flash Attention MAPE 根因的含义

当前实验已直接证实三类误差源：

1. [COMPUTED] **measurement contract mismatch**：eager、single graph 和 steady-state
   throughput 不是同一标签，小 workload 差异可达数微秒。
2. [COMPUTED] **resource extraction omission**：只看 binary 静态 shared 会漏掉 Triton
   launch 的动态 shared，进而错误估算 occupancy。
3. [COMPUTED] **feature/sample imbalance**：未压缩的多层 IR histogram 在小数据上严重
   过拟合，增加特征反而提高 MAPE。

仍未由本实验直接归因的部分：

- [FRAME] 生产 FlashAttention/FlashInfer 的真实 kernel selection、split-KV、merge、
  LSE/output writeback 和多 kernel launch 序列；
- [FRAME] PipeWeave 当前 decomposer/scheduler 与生产实现之间的 config/grid 差异；
- [FRAME] A100 与 Blackwell 的 architecture transfer；
- [FRAME] 编译版本、dtype、ragged/paged KV 和 decode/prefill 工作负载覆盖。

[INFERRED] 因而下一阶段不应直接重训大 MLP。应先把生产 FA 的真实 plan 注入为 oracle，
分别固定 decomposition、schedule、resource 和 launch-overhead，观察每次干预后的 MAPE；
只有剩余残差再交给新数据和模型。

## Go/No-Go

- [COMPUTED] **GO：** 继续 NVIDIA 多层 IR 路线；capture coverage、correctness、artifact
  determinism 和 graph 标签稳定性均通过第一周门槛。
- [COMPUTED] **GO：** 使用 compact TTGIR+target-IR feature 做下一轮消融。
- [FRAME] **NO-GO：** 暂不宣称生产 FA 的主要误差已经定量分解。
- [FRAME] **NO-GO：** A100 到位和 production kernel 接入前，不开始 AMD/Intel 后端实现，
  也不把 5090 单架构结果作为 CCF-A 级证据。

## 下一轮优先级

1. [FRAME] 接入生产 FlashAttention/FlashInfer Triton kernel 与真实 launch plan；保存每个
   子 kernel 的 case ID，并构建 aggregate latency record。
2. [FRAME] 建立 oracle intervention：真实 split count/grid、真实 tile/warp/stage、真实
   dynamic shared/register、真实 merge/launch 固定开销，逐项替换并报告 ΔMAPE。
3. [FRAME] 对每种 grouped split 运行至少 10 个 seed，并加入 bootstrap 置信区间和 alpha
   选择，避免单一 holdout 偶然性。
4. [FRAME] A100 到位后保持相同 source/case/schema，执行 architecture holdout；CUDA 13+
   作为 5090 主标签环境，A100 使用其官方支持工具链并单独记录 compiler ID。
5. [FRAME] 扩充 decode/prefill、causal/non-causal、D=64/128/256、ragged/paged KV、
   split-KV 与小 batch 固定开销 case。

## 复现实验

- 生成矩阵：python -m pipeweave_ir generate --profile rtx5090-day6 --out RUN_SUITE.json
- 采集：python -m pipeweave_ir capture --suite RUN_SUITE.json --out RUN_DIR
  --require-cuda-major 13 --cuda-bin /usr/local/cuda-13.2/bin
- 特征：python -m pipeweave_ir features --artifacts RUN_DIR/artifacts
  --out RUN_DIR/features.jsonl
- 数据集：python -m pipeweave_ir dataset --run RUN_DIR --out RUN_DIR/dataset.jsonl

[COMPUTED] 原始 IR、binary、measurement 与 ablation JSON 保存在 5090 的
/home/chen/pipeweave-ir-mvp/runs/；它们按项目策略不进入 Git。
[COMPUTED] 本报告关键结论置信度：artifact/正确性 **高**，重复性 **高**，当前 MAPE
消融 **低到中**，production FA 根因外推 **低**。
