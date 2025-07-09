#!/bin/bash
device="cuda"

echo "=========================================="
echo "Starting Chronos Evaluation Script"
echo "Device: $device"
echo "=========================================="
echo ""

conda activate reps-tsfm

mkdir -p results/quantitative_pruning/chronos

echo "[1/6] Evaluating vanilla Chronos T5-Large model..."
echo "Model: amazon/chronos-t5-large"
echo "Output: results/quantitative_pruning/chronos/vanilla-zero-shot.csv"
python chronos-forecasting/scripts/evaluation/evaluate.py chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml results/quantitative_pruning/chronos/vanilla-zero-shot.csv \
    --chronos-model-id "amazon/chronos-t5-large" \
    --batch-size=32 \
    --device $device \
    --num-samples 20
echo "Vanilla model evaluation completed"
echo ""

echo "[2/6] Evaluating Chronos with all blocks skipped..."
echo "Model: skip_blocks_results/models/chronos_skipped_all"
echo "Output: results/quantitative_pruning/chronos/skipped_all-zero-shot.csv"
python chronos-forecasting/scripts/evaluation/evaluate.py chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml results/quantitative_pruning/chronos/skipped_all-zero-shot.csv \
    --chronos-model-id "skip_blocks_results/models/chronos_skipped_all" \
    --batch-size=32 \
    --device $device \
    --num-samples 20
echo "All blocks skipped evaluation completed"
echo ""

echo "[3/6] Evaluating Chronos with block 1 skipped..."
echo "Model: skip_blocks_results/models/chronos_skipped_block1"
echo "Output: results/quantitative_pruning/chronos/skipped_block1-zero-shot.csv"
python chronos-forecasting/scripts/evaluation/evaluate.py chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml results/quantitative_pruning/chronos/skipped_block1-zero-shot.csv \
    --chronos-model-id "skip_blocks_results/models/chronos_skipped_block1" \
    --batch-size=32 \
    --device $device \
    --num-samples 20
echo "Block 1 skipped evaluation completed"
echo ""

echo "[4/6] Evaluating Chronos with block 2 skipped..."
echo "Model: skip_blocks_results/models/chronos_skipped_block2"
echo "Output: results/quantitative_pruning/chronos/skipped_block2-zero-shot.csv"
python chronos-forecasting/scripts/evaluation/evaluate.py chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml results/quantitative_pruning/chronos/skipped_block2-zero-shot.csv \
    --chronos-model-id "skip_blocks_results/models/chronos_skipped_block2" \
    --batch-size=32 \
    --device $device \
    --num-samples 20
echo "Block 2 skipped evaluation completed"
echo ""

echo "[5/6] Evaluating Chronos with block 3 skipped..."
echo "Model: skip_blocks_results/models/chronos_skipped_block3"
echo "Output: results/quantitative_pruning/chronos/skipped_block3-zero-shot.csv"
python chronos-forecasting/scripts/evaluation/evaluate.py chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml results/quantitative_pruning/chronos/skipped_block3-zero-shot.csv \
    --chronos-model-id "skip_blocks_results/models/chronos_skipped_block3" \
    --batch-size=32 \
    --device $device \
    --num-samples 20
echo "Block 3 skipped evaluation completed"
echo ""

echo "[6/6] Evaluating Chronos with block 4 skipped..."
echo "Model: skip_blocks_results/models/chronos_skipped_block4"
echo "Output: results/quantitative_pruning/chronos/skipped_block4-zero-shot.csv"
python chronos-forecasting/scripts/evaluation/evaluate.py chronos-forecasting/scripts/evaluation/configs/zero-shot.yaml results/quantitative_pruning/chronos/skipped_block4-zero-shot.csv \
    --chronos-model-id "skip_blocks_results/models/chronos_skipped_block4" \
    --batch-size=32 \
    --device $device \
    --num-samples 20
echo "Block 4 skipped evaluation completed"
echo ""

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in results/quantitative_pruning/chronos/ directory"
echo "=========================================="