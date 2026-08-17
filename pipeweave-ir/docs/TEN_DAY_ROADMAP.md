# 未来十个工作日实现路线

## 总体输出

- [FRAME] 一套 NVIDIA 多层 IR 捕获与版本化 artifact pipeline。
- [FRAME] 一套 RTX 5090 correctness/timing 数据采集流程。
- [FRAME] TTGIR-only、PTX-only、binary-only 和 combined 的静态 feature table。
- [FRAME] 一份首轮 5090 报告，以及进入大规模采集前的 Go/No-Go 结论。

## Day 1–5

[FRAME] 详细任务见 [WEEK1_EXECUTION_PLAN.md](WEEK1_EXECUTION_PLAN.md)。

## Day 6：扩充 case 生成器

- [x] [COMPUTED] 将 GEMM 扩展到小 M、方阵、大 K、宽 N 和非整 tile 边界。
- [x] [COMPUTED] 将 attention 扩展到 N=128/256/512/1024、head dimension 64/128、causal/non-causal。
- [x] [COMPUTED] 每个 problem 生成 4–8 个合法 num_warps/num_stages/tile config。
- [ ] [FRAME] 增加非法/OOM config 的显式状态，不把它们混入 latency 样本。

## Day 7：重复性和 compiler-version 基线

- [x] [COMPUTED] 相同 suite 连续运行三次。
- [x] [COMPUTED] 比较 canonical IR hash、binary hash、latency CV 和 clock/temperature 漂移。
- [ ] [FRAME] 如条件允许，用 Triton 3.4/CUDA 12.8 只做 compiler-drift 对照，不作为主性能标签。
- [x] [COMPUTED] 主 Blackwell 性能数据继续只使用 CUDA 13.0+。

## Day 8：第一版消融数据集

- [x] [COMPUTED] 合并 manifest、features 和 measurement，输出一行一个 case/mode 的 JSONL/Parquet-ready 表。
- [x] [COMPUTED] 建立 shape-holdout 和 config-holdout；禁止随机逐行泄漏。
- [x] [COMPUTED] 实现简单线性/ridge baseline，暂不引入复杂 GNN。
- [x] [COMPUTED] 输出 TTGIR-only、PTX-only、binary-only、combined 的训练/测试指标。

## Day 9：配置排序与 oracle 接口

- [x] [COMPUTED] 实现同一 problem 内的 Spearman、top-k recall 和 best-config regret。
- [ ] [FRAME] 支持将真实 grid、warps、stages、resources 注入为 oracle feature。
- [x] [COMPUTED] 将绝对 latency 误差与 ranking 误差分开报告。
- [ ] [FRAME] 选择至少 5 个 lowering 差异 case 做人工核对。

## Day 10：阶段评审与发布

- [ ] [FRAME] 生成 `reports/RTX5090_TEN_DAY_REVIEW.md`。
- [ ] [FRAME] 回答 TTGIR 与目标 IR 是否具有独立增益。
- [x] [COMPUTED] 确定下一阶段是扩大 NVIDIA 数据、先补生产 FA，还是转向 ranking cost model。
- [ ] [FRAME] 运行 tests/lint，提交本次新目录并推送 GitHub 草稿 PR。
- [x] [COMPUTED] A100 到位前保留 architecture-holdout 接口，不用两个 SM120 SKU 代替跨代结论。

## 十日阶段验收指标

- [FRAME] artifact capture coverage 目标：至少 95%。
- [FRAME] correctness pass rate 目标：至少 95%。
- [FRAME] `>=20 µs` case 的重复测量 CV 目标：不高于 3%；更小 case 单独报告。
- [FRAME] exhaustive 子集具备 top-k recall 和 regret 结果。
- [FRAME] combined feature 是否优于最佳单层 feature 必须有定量结论；若没有，不启动 AMD/Intel 开发。
