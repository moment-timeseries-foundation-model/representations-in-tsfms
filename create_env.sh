#!/bin/bash

set -e

echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Properly source conda into the current shell
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reps-tsfm

pip install momentfm==0.1.4

echo "Environment created successfully!"
echo "To activate the environment, run:"
echo "conda activate reps-tsfm"
echo "Remember to run all experiments from the reps-tsfm environment!"