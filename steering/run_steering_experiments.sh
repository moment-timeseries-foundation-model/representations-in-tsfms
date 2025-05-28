#!/bin/bash

# Set up error handling
set -e
echo "Starting steering experiments..."

# Create directories for results if they don't exist
mkdir -p results/sine_steering
mkdir -p results/trend_steering
mkdir -p results/compositional_steering

# ============================================================================
# 1. Steering constant signals to sinusoidal (periodic) outputs
# ============================================================================
echo "1. Running sinusoidal steering experiments..."

# MOMENT model
echo "  1.1 Running sinusoidal steering with MOMENT model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/sine_constant.parquet \
  --input-sample datasets/none_constant.parquet \
  --model moment \
  --method mean \
  --alpha 1.0 \
  --output-dir results/sine_steering/moment

# Chronos model
echo "  1.2 Running sinusoidal steering with Chronos model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/sine_constant.parquet \
  --input-sample datasets/none_constant.parquet \
  --model chronos \
  --method mean \
  --alpha 1.0 \
  --output-dir results/sine_steering/chronos

# ============================================================================
# 2. Steering constant signals to increasing or decreasing trend
# ============================================================================
echo "2. Running trend steering experiments..."

# MOMENT model - Increasing trend
echo "  2.1 Running increasing trend steering with MOMENT model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_increasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model moment \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/moment_increasing

# MOMENT model - Decreasing trend
echo "  2.2 Running decreasing trend steering with MOMENT model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_decreasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model moment \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/moment_decreasing

# Chronos model - Increasing trend
echo "  2.3 Running increasing trend steering with Chronos model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_increasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model chronos \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/chronos_increasing

# Chronos model - Decreasing trend
echo "  2.4 Running decreasing trend steering with Chronos model..."
steertool steer \
  --source-dataset datasets/none_constant.parquet \
  --target-dataset datasets/none_decreasing.parquet \
  --input-sample datasets/none_constant.parquet \
  --model chronos \
  --method mean \
  --alpha 1.0 \
  --output-dir results/trend_steering/chronos_decreasing

# ============================================================================
# 3. Compositional steering — combining trend and periodicity
# ============================================================================
echo "3. Running compositional steering experiments..."

# MOMENT model - composition with different beta values
echo "  3.1 Running compositional steering with MOMENT model..."
for beta in 0.3 0.5 0.7
do
  # Combine sinusoidal and increasing trend
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

  # Combine sinusoidal and decreasing trend
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

# Chronos model - composition with different beta values
echo "  3.2 Running compositional steering with Chronos model..."
for beta in 0.3 0.5 0.7
do
  # Combine sinusoidal and increasing trend
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

  # Combine sinusoidal and decreasing trend
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