#!/bin/bash

set -e
echo "Starting steering experiments..."

mkdir -p results/sine_steering
mkdir -p results/trend_steering
mkdir -p results/compositional_steering

echo "1. Running sinusoidal steering experiments..."

echo "  1.1 Running sinusoidal steering with MOMENT model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/sine_constant.parquet \
  --input-sample datasets/none_constant.parquet \
  --model moment \
  --method mean \
  --alpha 1.0 \
  --output-dir results/sine_steering/moment

echo "  1.2 Running sinusoidal steering with Chronos model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/sine_constant.parquet \
  --input-sample datasets/none_constant.parquet \
  --model chronos \
  --method mean \
  --alpha 1.0 \
  --output-dir results/sine_steering/chronos

echo "2. Running trend steering experiments..."

echo "  2.1 Running increasing trend steering with MOMENT model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_increasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model moment \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/moment_increasing

echo "  2.2 Running decreasing trend steering with MOMENT model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_decreasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model moment \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/moment_decreasing

echo "  2.3 Running increasing trend steering with Chronos model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_increasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model chronos \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/chronos_increasing

echo "  2.4 Running decreasing trend steering with Chronos model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_decreasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model chronos \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/chronos_decreasing

echo "3. Running compositional steering experiments..."

echo "  3.1 Running compositional steering with MOMENT model..."
for beta in 0.3 0.5 0.7
do
  steertool steer \
    --source-dataset datasets/none_constant.parquet \
    --target-dataset datasets/sine_constant.parquet \
    --second-target-dataset datasets/none_increasing.parquet \
    --input-sample datasets/none_constant.parquet \
    --model moment \
    --method mean \
    --alpha 1.0 \
    --beta $beta \
    --output-dir results/compositional_steering/moment_sine_increasing_beta${beta}

  steertool steer \
    --source-dataset datasets/none_constant.parquet \
    --target-dataset datasets/sine_constant.parquet \
    --second-target-dataset datasets/none_decreasing.parquet \
    --input-sample datasets/none_constant.parquet \
    --model moment \
    --method mean \
    --alpha 1.0 \
    --beta $beta \
    --output-dir results/compositional_steering/moment_sine_decreasing_beta${beta}
done

echo "  3.2 Running compositional steering with Chronos model..."
for beta in 0.3 0.5 0.7
do
  steertool steer \
    --source-dataset datasets/none_constant.parquet \
    --target-dataset datasets/sine_constant.parquet \
    --second-target-dataset datasets/none_increasing.parquet \
    --input-sample datasets/none_constant.parquet \
    --model chronos \
    --method mean \
    --alpha 1.0 \
    --beta $beta \
    --output-dir results/compositional_steering/chronos_sine_increasing_beta${beta}

  steertool steer \
    --source-dataset datasets/none_constant.parquet \
    --target-dataset datasets/sine_constant.parquet \
    --second-target-dataset datasets/none_decreasing.parquet \
    --input-sample datasets/none_constant.parquet \
    --model chronos \
    --method mean \
    --alpha 1.0 \
    --beta $beta \
    --output-dir results/compositional_steering/chronos_sine_decreasing_beta${beta}
done

echo "All experiments completed successfully!"
echo "Results are saved in the 'results/' directory." 