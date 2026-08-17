# PipeWeave-IR

[FRAME] PipeWeave-IR is the NVIDIA-first MVP for studying whether target-specific TTGIR and backend lowering artifacts can jointly improve Triton kernel performance prediction.

[KNOWN] The first validation target is an NVIDIA GeForce RTX 5090 running a CUDA 13.2 PyTorch environment. [FRAME] The project keeps measurement labels separate from compile-only artifacts so later experiments can distinguish zero-shot, architecture-calibrated, and label-trained modes.

## Initial scope

- [FRAME] Capture TTIR, TTGIR, LLVM IR, PTX, cubin/SASS, launch metadata, and final binary resource reports.
- [FRAME] Benchmark controlled Triton GEMM and Flash-Attention-like kernels on RTX 5090.
- [FRAME] Store raw timing samples and explicit eager/graph measurement modes.
- [FRAME] Extract interpretable TTGIR, PTX, and binary-resource features.
- [FRAME] Join grouped datasets and compare compact/raw multi-level IR ridge ablations.
- [FRAME] Keep eager, single-node CUDA Graph, and batched-graph throughput as separate labels.

## Repository layout

```text
pipeweave-ir/
  configs/                 Reproducible case suites
  docs/                    Week-one plan, ten-day roadmap, and schema
  scripts/                 Remote 5090 runner
  src/pipeweave_ir/        Capture, benchmark, feature, and metric code
  tests/                   CPU-only unit tests
  reports/                 Versioned experiment summaries
  artifacts/               Generated compiler outputs; ignored by Git
  runs/                    Generated measurements; ignored by Git
```

## RTX 5090 quick start

[KNOWN] The current 5090 host has a `drkernel-vllm027-cu132` conda environment with CUDA 13.2 and Triton 3.7.1.

```bash
export PIPEWEAVE_IR_PYTHON=/home/chen/anaconda3/envs/drkernel-vllm027-cu132/bin/python
export PYTHONPATH=$PWD/src

$PIPEWEAVE_IR_PYTHON -m pipeweave_ir env \
  --require-cuda-major 13 \
  --out runs/rtx5090/environment.json

$PIPEWEAVE_IR_PYTHON -m pipeweave_ir capture \
  --suite configs/rtx5090_smoke.json \
  --out runs/rtx5090 \
  --require-cuda-major 13

$PIPEWEAVE_IR_PYTHON -m pipeweave_ir features \
  --artifacts runs/rtx5090/artifacts \
  --out runs/rtx5090/features.jsonl
```

## Validation

```bash
pytest
ruff check .
```

[FRAME] GPU profilers may be used on a small validation subset, but profiler counters are not predictor inputs. [FRAME] Generated cubins and raw runs remain outside Git; manifests, schemas, code, and summarized reports are versioned.

## Day 6–9 commands

- Generate 89 cases: python -m pipeweave_ir generate --profile rtx5090-day6 --out runs/day6.json
- Join labels: python -m pipeweave_ir dataset --run runs/rtx5090 --out runs/rtx5090/dataset.jsonl
- Run grouped ablation: python -m pipeweave_ir ablate --dataset runs/rtx5090/dataset.jsonl --out runs/rtx5090/ablation.json --split-by shape
- Compare repeated runs: python -m pipeweave_ir repeatability --run runs/r1 --run runs/r2 --run runs/r3 --out runs/repeatability.json

## Plans

- [Week 1 execution plan](docs/WEEK1_EXECUTION_PLAN.md)
- [Ten-day implementation roadmap](docs/TEN_DAY_ROADMAP.md)
- [Artifact and measurement schema](docs/DATA_SCHEMA.md)
- [Hardware and data requirements](docs/REQUIREMENTS.md)
- [RTX 5090 Week 1 report](reports/RTX5090_WEEK1.md)
