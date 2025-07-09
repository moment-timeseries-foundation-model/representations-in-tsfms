#!/bin/bash

mkdir -p results/separability_analysis

MODEL="moment"
SAMPLES=20
DEVICE="cuda"
RESULTS_DIR="results/separability_analysis"

echo "======================================================================================"
echo "Running Separability Analysis for Time Series Patterns"
echo "======================================================================================"

echo "Running analysis: Constant vs. Sinusoidal"
python -m steertool.cli analyze \
    --dataset1 ./datasets/none_constant.parquet \
    --dataset2 ./datasets/sine_constant.parquet \
    --type constant-sine \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/constant_vs_sine \
    --device $DEVICE

echo "Running analysis: Increasing vs. Decreasing Trends"
python -m steertool.cli analyze \
    --dataset1 ./datasets/none_increasing.parquet \
    --dataset2 ./datasets/none_decreasing.parquet \
    --type trend \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/increasing_vs_decreasing \
    --device $DEVICE

echo "Running analysis: High vs. Low Periodicity"
python -m steertool.cli analyze \
    --dataset1 ./datasets/sine_freq_high.parquet \
    --dataset2 ./datasets/sine_freq_low.parquet \
    --type periodicity \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/high_vs_low_frequency \
    --device $DEVICE

echo "Running analysis: Constant vs. Trend (Increasing)"
python -m steertool.cli analyze \
    --dataset1 ./datasets/none_constant.parquet \
    --dataset2 ./datasets/none_increasing.parquet \
    --type constant-sine \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/constant_vs_trend_increasing \
    --device $DEVICE

echo "Running analysis: Constant vs. Trend (Decreasing)"
python -m steertool.cli analyze \
    --dataset1 ./datasets/none_constant.parquet \
    --dataset2 ./datasets/none_decreasing.parquet \
    --type constant-sine \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/constant_vs_trend_decreasing \
    --device $DEVICE

echo "Running analysis: Sinusoidal vs. Trend (Increasing)"
python -m steertool.cli analyze \
    --dataset1 ./datasets/sine_constant.parquet \
    --dataset2 ./datasets/sine_increasing.parquet \
    --type trend \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/sine_vs_trend_increasing \
    --device $DEVICE

echo "Running analysis: Sinusoidal vs. Trend (Decreasing)"
python -m steertool.cli analyze \
    --dataset1 ./datasets/sine_constant.parquet \
    --dataset2 ./datasets/sine_decreasing.parquet \
    --type trend \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/sine_vs_trend_decreasing \
    --device $DEVICE

echo "Running analysis: High vs. Low Amplitude (Sine)"
python -m steertool.cli analyze \
    --dataset1 ./datasets/sine_amp_high.parquet \
    --dataset2 ./datasets/sine_amp_low.parquet \
    --type periodicity \
    --model $MODEL \
    --samples $SAMPLES \
    --output-dir $RESULTS_DIR/high_vs_low_amplitude \
    --device $DEVICE

echo "======================================================================================"
echo "Separability analysis complete. Results are saved in $RESULTS_DIR"
echo "======================================================================================"

echo "Generated visualizations:"
for dir in $RESULTS_DIR/*; do
    count=$(find "$dir" -name "*.pdf" | wc -l)
    echo "- $(basename $dir): $count visualizations"
done 