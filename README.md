# Representations in Time Series Foundation Models

This repository contains the code for our ICML '25 paper: "Exploring Representations and Interventions in Time Series Foundation Models".

## Overview

Time series foundation models (TSFMs) are powerful tools for various applications, but their internal representations and learned concepts are not well understood. In this study, we:

1. **Analyze representation similarity**: Investigate the structure and redundancy of representations across various TSFMs
2. **Perform model pruning**: Leverage redundancy in representations to prune layers and improve efficiency
3. **Identify and localize concepts**: Explore what concepts (periodicity, trends) are learned by these models
4. **Implement concept steering**: Manipulate latent space to influence model behavior

## Repository Structure

```
representations-in-tsfms/
├── efficiency/                # Code for representation analysis and pruning
│   ├── chronos-forecasting/   # Chronos model implementation
│   ├── tsfm_similarity/       # Similarity analysis tools
│   ├── produce_similarity_maps.sh
│   ├── produce_and_time_models.sh
│   └── evaluate_chronos_variants.sh
├── steering/                  # Code for concept identification and steering
│   ├── chronos_notebooks/     # Notebooks for Chronos model
│   ├── chronos_viz/           # Visualization for Chronos steering
│   ├── moment_notebooks/      # Notebooks for MOMENT model
│   ├── moment_viz/            # Visualization for MOMENT steering
│   ├── src/                   # Core steering implementation
│   └── datasets/              # Datasets for steering experiments
└── environment.yml            # Conda environment specification
```

## Installation

1. Clone the repository and initialize submodules:
```bash
git clone --recurse-submodules git@github.com:moment-timeseries-foundation-model/representations-in-tsfms.git
cd representations-in-tsfms
```

2. Make sure that you have `conda` installed and create the environment:
```bash
bash create_env.sh
conda activate reps-tsfm
```

## Experiments

### Representation Analysis and Pruning

Analyze model representation similarity and prune redundant layers in TSFMs:

```bash
# Generate similarity maps between layers of TSFMs
cd efficiency
./produce_similarity_maps.sh

# Produce pruned models and time them
./produce_and_time_models.sh

# Evaluate pruned Chronos model variants
./evaluate_chronos_variants.sh
```

Results will be available in the `results` directory.

### Concept Identification and Steering (Coming Soon)

Identify concepts in TSFMs and perform concept steering:

```bash
# Example commands will be provided when this section is completed
cd steering
# Run concept identification
# Run concept steering
```

## Citation (TODO)

```bibtex
@inproceedings{representations-tsfms-2025,
  title={Exploring Representations and Interventions in Time Series Foundation Models},
  author={Your Name and Co-authors},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

## License (TODO)
This project is licensed under the terms of the LICENSE file included in this repository.
