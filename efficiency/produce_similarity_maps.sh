#!/bin/bash

conda activate reps-tsfm

config_files=(
    "random_univariate.yaml"
)

for config in "${config_files[@]}"; do
    echo "Running experiment with config: $config"
    python -m tsfm_similarity.experiments.similarity.similarity_experiment --config "tsfm_similarity/experiments/similarity/config/$config"
    echo "Finished experiment with config: $config"
    echo "----------------------------------------"
done

echo "All experiments completed."
