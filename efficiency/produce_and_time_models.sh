#!/bin/bash

DEVICE="cuda"
SAMPLES=40
SEQUENCE_LENGTH=512
BATCH_SIZE=1

mkdir -p results/pruning

echo "Starting Skip Blocks Evaluation"
echo "Device: $DEVICE"
echo "Samples: $SAMPLES"
echo "==========================================="

conda activate reps-tsfm

run_evaluation() {
    local model_type=$1
    local skip_name=$2
    local blocks_to_skip=$3
    local extra_args=$4
    
    echo ""
    echo "Running $model_type model - $skip_name configuration"
    echo "Blocks to skip: $blocks_to_skip"
    echo "-------------------------------------------"
    
    python -m tsfm_similarity.produce_models \
        --model_type $model_type \
        --skip_name $skip_name \
        --blocks_to_skip "$blocks_to_skip" \
        --num_inference_samples $SAMPLES \
        --batch_size $BATCH_SIZE \
        --sequence_length $SEQUENCE_LENGTH \
        --device $DEVICE \
        --save_model \
        $extra_args
}

run_baseline() {
    local model_type=$1
    
    echo ""
    echo "Running $model_type baseline (original model)"
    echo "-------------------------------------------"
    
    python -m tsfm_similarity.produce_models \
        --model_type $model_type \
        --no_skip \
        --num_inference_samples $SAMPLES \
        --batch_size $BATCH_SIZE \
        --sequence_length $SEQUENCE_LENGTH \
        --device $DEVICE \
        --save_model
}

echo ""
echo "=========================================="
echo "CHRONOS MODEL EVALUATIONS"
echo "=========================================="

run_baseline "chronos"

run_evaluation "chronos" "all" "2-3,6-8,11-12,16-21"
run_evaluation "chronos" "block1" "2-3"
run_evaluation "chronos" "block2" "6-8"
run_evaluation "chronos" "block3" "11-12"
run_evaluation "chronos" "block4" "16-21"

echo ""
echo "=========================================="
echo "MOMENT MODEL EVALUATIONS"
echo "=========================================="

run_baseline "moment"

run_evaluation "moment" "all" "2-4,10-17,20-22"
run_evaluation "moment" "block1" "2-4"
run_evaluation "moment" "block2" "10-17"
run_evaluation "moment" "block3" "20-22"

echo ""
echo "=========================================="
echo "DETAILED PROFILING RUNS"
echo "=========================================="

run_evaluation "chronos" "all_profiled" "2-3,6-8,11-12,16-21" "--profile"
run_evaluation "moment" "all_profiled" "2-4,10-17,20-22" "--profile"

echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
echo "Results saved in: results/pruning/"
echo "Check the JSON files for detailed metrics"
echo "Check the models/ subdirectory for saved weights"
echo "Check the traces/ subdirectory for profiling traces" 