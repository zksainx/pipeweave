#!/bin/bash

# tp8.sh - Run aggregator.py for Llama-3.1-70B (TP=8) workloads on H800 hardware
# Uses fa3 for Hopper architecture and includes collective communication

set -e

# Define workload paths
ARXIV_16="workload/Llama-3.1-70B_arxiv_16_fa3_tp8_pp1.json"
ARXIV_32="workload/Llama-3.1-70B_arxiv_32_fa3_tp8_pp1.json"
SPLITWISE_64="workload/Llama-3.1-70B_splitwise_64_fa3_tp8_pp1.json"
SPLITWISE_72="workload/Llama-3.1-70B_splitwise_72_fa3_tp8_pp1.json"

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

# Hopper architecture GPU (H800) - use fa3
echo "Processing H800 (Hopper architecture) with TP=8..."

# H800 - arxiv_16 workload
run_aggregator "H800" "$ARXIV_16" "Llama-3.1-70B_arxiv_16_fa3_tp8_pp1_H800.json" "H800"

# H20 - arxiv_16 workload
run_aggregator "H20" "$ARXIV_16" "Llama-3.1-70B_arxiv_16_fa3_tp8_pp1_H20.json" "H20"

# # H800 - arxiv_32 workload
# run_aggregator "H800" "$ARXIV_32" "Llama-3.1-70B_arxiv_32_fa3_tp8_pp1_H800.json" "H800"

# H800 - splitwise_64 workload
run_aggregator "H800" "$SPLITWISE_64" "Llama-3.1-70B_splitwise_64_fa3_tp8_pp1_H800.json" "H800"

# H20 - splitwise_64 workload
run_aggregator "H20" "$SPLITWISE_64" "Llama-3.1-70B_splitwise_64_fa3_tp8_pp1_H20.json" "H20"

# # H800 - splitwise_72 workload
# run_aggregator "H800" "$SPLITWISE_72" "Llama-3.1-70B_splitwise_72_fa3_tp8_pp1_H800.json" "H800"


echo "========================================"
echo "All predictions completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
