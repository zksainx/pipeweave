#!/bin/bash

# tp1.sh - Run aggregator.py for Qwen2.5-14B (TP=1) workloads on all hardware
# Automatically selects fa2 or fa3 based on hardware architecture

set -e

# Define workload base paths
ARXIV_FA2="workload/Qwen2.5-14B_arxiv_8_fa2_tp1_pp1.json"
ARXIV_FA3="workload/Qwen2.5-14B_arxiv_8_fa3_tp1_pp1.json"
SPLITWISE_FA2="workload/Qwen2.5-14B_splitwise_32_fa2_tp1_pp1.json"
SPLITWISE_FA3="workload/Qwen2.5-14B_splitwise_32_fa3_tp1_pp1.json"

# Output directory
OUTPUT_DIR="e2e/pipeweave_pred"
mkdir -p "$OUTPUT_DIR"

# Function to run aggregator for a given hardware and workload
run_aggregator() {
    local hardware=$1
    local workload=$2
    local output_name=$3

    echo "========================================"
    echo "Processing: $hardware - $output_name"
    echo "========================================"

    python3 aggregator.py \
        --workload "$workload" \
        --hardware "$hardware" \
        --model_dir mlp_models \
        --dataset_dir dataset \
        --hardware_dir hardware \
        --output "$OUTPUT_DIR/$output_name"

    echo "Completed: $output_name"
    echo ""
}

# Hopper architecture GPUs (use fa3)
echo "Processing Hopper architecture GPUs..."

# H100
run_aggregator "H100" "$ARXIV_FA3" "Qwen2.5-14B_arxiv_8_fa3_tp1_pp1_H100.json"
run_aggregator "H100" "$SPLITWISE_FA3" "Qwen2.5-14B_splitwise_32_fa3_tp1_pp1_H100.json"

# H20
run_aggregator "H20" "$ARXIV_FA3" "Qwen2.5-14B_arxiv_8_fa3_tp1_pp1_H20.json"
run_aggregator "H20" "$SPLITWISE_FA3" "Qwen2.5-14B_splitwise_32_fa3_tp1_pp1_H20.json"

# H200
run_aggregator "H200" "$ARXIV_FA3" "Qwen2.5-14B_arxiv_8_fa3_tp1_pp1_H200.json"
run_aggregator "H200" "$SPLITWISE_FA3" "Qwen2.5-14B_splitwise_32_fa3_tp1_pp1_H200.json"

# H800
run_aggregator "H800" "$ARXIV_FA3" "Qwen2.5-14B_arxiv_8_fa3_tp1_pp1_H800.json"
run_aggregator "H800" "$SPLITWISE_FA3" "Qwen2.5-14B_splitwise_32_fa3_tp1_pp1_H800.json"

# Ampere architecture GPUs (use fa2)
echo "Processing Ampere architecture GPUs..."

# A100
run_aggregator "A100" "$ARXIV_FA2" "Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_A100.json"
run_aggregator "A100" "$SPLITWISE_FA2" "Qwen2.5-14B_splitwise_32_fa2_tp1_pp1_A100.json"

# A40
run_aggregator "A40" "$ARXIV_FA2" "Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_A40.json"
run_aggregator "A40" "$SPLITWISE_FA2" "Qwen2.5-14B_splitwise_32_fa2_tp1_pp1_A40.json"

# RTX A6000
run_aggregator "RTX A6000" "$ARXIV_FA2" "Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_RTX_A6000.json"
run_aggregator "RTX A6000" "$SPLITWISE_FA2" "Qwen2.5-14B_splitwise_32_fa2_tp1_pp1_RTX_A6000.json"

# Ada architecture GPUs (use fa2)
echo "Processing Ada architecture GPUs..."

# L20
run_aggregator "L20" "$ARXIV_FA2" "Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_L20.json"
run_aggregator "L20" "$SPLITWISE_FA2" "Qwen2.5-14B_splitwise_32_fa2_tp1_pp1_L20.json"

# L40
run_aggregator "L40" "$ARXIV_FA2" "Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_L40.json"
run_aggregator "L40" "$SPLITWISE_FA2" "Qwen2.5-14B_splitwise_32_fa2_tp1_pp1_L40.json"

# RTX 6000 Ada
run_aggregator "RTX 6000 Ada" "$ARXIV_FA2" "Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_RTX_6000_Ada.json"
run_aggregator "RTX 6000 Ada" "$SPLITWISE_FA2" "Qwen2.5-14B_splitwise_32_fa2_tp1_pp1_RTX_6000_Ada.json"

# Blackwell architecture GPUs (use fa2 for now, as it's not explicitly hopper)
echo "Processing Blackwell architecture GPUs..."

# RTX PRO 6000 S
run_aggregator "RTX PRO 6000 S" "$ARXIV_FA2" "Qwen2.5-14B_arxiv_8_fa2_tp1_pp1_RTX_PRO_6000_S.json"
run_aggregator "RTX PRO 6000 S" "$SPLITWISE_FA2" "Qwen2.5-14B_splitwise_32_fa2_tp1_pp1_RTX_PRO_6000_S.json"

echo "========================================"
echo "All predictions completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
