#!/bin/bash

# vllm_tp4_pp2.sh - Run aggregator.py for Llama-3.1-70B vLLM workloads (TP=4, PP=2) on H800 and H20 hardware
# Uses fa3 for Hopper architecture and includes collective communication

set -e


# Define workload paths
ARXIV_16="workload/Llama-3.1-70B_vllm_arxiv16_fa3_tp4_pp2.json"
SPLITWISE_64="workload/Llama-3.1-70B_vllm_splitwise64_fa3_tp4_pp2.json"

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

# H800 hardware - arxiv_16 workload
echo "Processing H800 (Hopper architecture) with TP=4, PP=2..."

run_aggregator "H800" "$ARXIV_16" "Llama-3.1-70B_vllm_arxiv16_fa3_tp4_pp2_H800.json" "H800"

# H800 - splitwise_64 workload
run_aggregator "H800" "$SPLITWISE_64" "Llama-3.1-70B_vllm_splitwise64_fa3_tp4_pp2_H800.json" "H800"

# H20 hardware - arxiv_16 workload
echo "Processing H20 (Hopper architecture) with TP=4, PP=2..."

run_aggregator "H20" "$ARXIV_16" "Llama-3.1-70B_vllm_arxiv16_fa3_tp4_pp2_H20.json" "H20"

# H20 - splitwise_64 workload
run_aggregator "H20" "$SPLITWISE_64" "Llama-3.1-70B_vllm_splitwise64_fa3_tp4_pp2_H20.json" "H20"


echo "========================================"
echo "All vLLM TP=4 PP=2 predictions completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
