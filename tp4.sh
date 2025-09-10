#!/bin/bash

# tp4.sh - Run aggregator.py for Llama-3.1-70B (TP=4) workloads on A100 and H100 hardware
# Uses fa2 for Ampere, fa3 for Hopper architecture, and includes collective communication

set -e

# Define workload base paths
ARXIV_FA2="workload/Llama-3.1-70B_arxiv_16_fa2_tp4_pp1.json"
ARXIV_FA3="workload/Llama-3.1-70B_arxiv_16_fa3_tp4_pp1.json"
SPLITWISE_FA2="workload/Llama-3.1-70B_splitwise_64_fa2_tp4_pp1.json"
SPLITWISE_FA3="workload/Llama-3.1-70B_splitwise_64_fa3_tp4_pp1.json"

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
echo "Processing A100 (Ampere architecture) with TP=4..."

# A100 - arxiv workload
run_aggregator "A100" "$ARXIV_FA2" "Llama-3.1-70B_arxiv_16_fa2_tp4_pp1_A100.json" "A100"

# A100 - splitwise workload
run_aggregator "A100" "$SPLITWISE_FA2" "Llama-3.1-70B_splitwise_64_fa2_tp4_pp1_A100.json" "A100"

# Hopper architecture GPU (H100) - use fa3
echo "Processing H100 (Hopper architecture) with TP=4..."

# H100 - arxiv workload
run_aggregator "H100" "$ARXIV_FA3" "Llama-3.1-70B_arxiv_16_fa3_tp4_pp1_H100.json" "H100"

# H100 - splitwise workload
run_aggregator "H100" "$SPLITWISE_FA3" "Llama-3.1-70B_splitwise_64_fa3_tp4_pp1_H100.json" "H100"


echo "========================================"
echo "All predictions completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "========================================"
