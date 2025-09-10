#!/bin/bash

# tp2.sh - Run aggregator.py for Qwen3-32B (TP=2) workloads on A100 and H100 hardware
# Uses fa2 for Ampere, fa3 for Hopper architecture, and includes collective communication

set -e


# Define workload base paths
ARXIV_16_FA2="workload/Qwen3-32B_arxiv_16_fa2_tp2_pp1.json"
ARXIV_16_FA3="workload/Qwen3-32B_arxiv_16_fa3_tp2_pp1.json"
ARXIV_12_FA2="workload/Qwen3-32B_arxiv_12_fa2_tp2_pp1.json"
ARXIV_12_FA3="workload/Qwen3-32B_arxiv_12_fa3_tp2_pp1.json"
SPLITWISE_64_FA2="workload/Qwen3-32B_splitwise_64_fa2_tp2_pp1.json"
SPLITWISE_64_FA3="workload/Qwen3-32B_splitwise_64_fa3_tp2_pp1.json"
SPLITWISE_48_FA2="workload/Qwen3-32B_splitwise_48_fa2_tp2_pp1.json"
SPLITWISE_48_FA3="workload/Qwen3-32B_splitwise_48_fa3_tp2_pp1.json"

# Output directory
OUTPUT_DIR="e2e/pipeweave_pred"
mkdir -p "$OUTPUT_DIR"

# Function to run aggregator for a given hardware and workload
run_aggregator() {
    local hardware=$1
    local workload=$2
    local output_name=$3
    local collective_hw=$4

    echo "========================================"
    echo "Processing: $hardware - $output_name"
    echo "========================================"

    python3 aggregator.py \
        --workload "$workload" \
        --hardware "$hardware" \
        --collective_hardware "$collective_hw" \
        --model_dir mlp_models \
        --dataset_dir dataset \
        --hardware_dir hardware \
        --output "$OUTPUT_DIR/$output_name"

    echo "Completed: $output_name"
    echo ""
}

# Ampere architecture GPU (A100) - use fa2
echo "Processing A100 (Ampere architecture) with TP=2..."

# A100 - arxiv_16 workload
run_aggregator "A100" "$ARXIV_16_FA2" "Qwen3-32B_arxiv_16_fa2_tp2_pp1_A100.json" "A100"

# A100 - arxiv_12 workload
run_aggregator "A100" "$ARXIV_12_FA2" "Qwen3-32B_arxiv_12_fa2_tp2_pp1_A100.json" "A100"

# A100 - splitwise_64 workload
run_aggregator "A100" "$SPLITWISE_64_FA2" "Qwen3-32B_splitwise_64_fa2_tp2_pp1_A100.json" "A100"

# A100 - splitwise_48 workload
run_aggregator "A100" "$SPLITWISE_48_FA2" "Qwen3-32B_splitwise_48_fa2_tp2_pp1_A100.json" "A100"

# Hopper architecture GPU (H100) - use fa3
echo "Processing H100 (Hopper architecture) with TP=2..."

# H100 - arxiv_16 workload
run_aggregator "H100" "$ARXIV_16_FA3" "Qwen3-32B_arxiv_16_fa3_tp2_pp1_H100.json" "H100"

# H100 - arxiv_12 workload
run_aggregator "H100" "$ARXIV_12_FA3" "Qwen3-32B_arxiv_12_fa3_tp2_pp1_H100.json" "H100"

# H100 - splitwise_64 workload
run_aggregator "H100" "$SPLITWISE_64_FA3" "Qwen3-32B_splitwise_64_fa3_tp2_pp1_H100.json" "H100"

# H100 - splitwise_48 workload
run_aggregator "H100" "$SPLITWISE_48_FA3" "Qwen3-32B_splitwise_48_fa3_tp2_pp1_H100.json" "H100"

# Blackwell architecture GPU (RTX PRO 6000 S, cc 12.0) - use fa2
echo "Processing RTX PRO 6000 S (Blackwell architecture) with TP=2..."

# RTX PRO 6000 S - arxiv_16 workload
run_aggregator "RTX PRO 6000 S" "$ARXIV_16_FA2" "Qwen3-32B_arxiv_16_fa2_tp2_pp1_RTX_PRO_6000_S.json" "RTX PRO 6000 S"

# RTX PRO 6000 S - arxiv_12 workload
run_aggregator "RTX PRO 6000 S" "$ARXIV_12_FA2" "Qwen3-32B_arxiv_12_fa2_tp2_pp1_RTX_PRO_6000_S.json" "RTX PRO 6000 S"

# RTX PRO 6000 S - splitwise_64 workload
run_aggregator "RTX PRO 6000 S" "$SPLITWISE_64_FA2" "Qwen3-32B_splitwise_64_fa2_tp2_pp1_RTX_PRO_6000_S.json" "RTX PRO 6000 S"

# RTX PRO 6000 S - splitwise_48 workload
run_aggregator "RTX PRO 6000 S" "$SPLITWISE_48_FA2" "Qwen3-32B_splitwise_48_fa2_tp2_pp1_RTX_PRO_6000_S.json" "RTX PRO 6000 S"

# Ada Lovelace architecture GPU (RTX 6000 Ada) - use fa2
echo "Processing RTX 6000 Ada (Ada Lovelace architecture) with TP=2..."

# RTX 6000 Ada - arxiv_12 workload
run_aggregator "RTX 6000 Ada" "$ARXIV_12_FA2" "Qwen3-32B_arxiv_12_fa2_tp2_pp1_RTX_6000_Ada.json" "RTX 6000 Ada"

# RTX 6000 Ada - splitwise_48 workload
run_aggregator "RTX 6000 Ada" "$SPLITWISE_48_FA2" "Qwen3-32B_splitwise_48_fa2_tp2_pp1_RTX_6000_Ada.json" "RTX 6000 Ada"

echo "========================================"
echo "All predictions completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
