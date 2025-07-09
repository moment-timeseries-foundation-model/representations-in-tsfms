# Representations in Time Series Foundation Models

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2409.12915&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2409.12915)
[![Python: 3.10](https://img.shields.io/badge/Python-3.10-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/license/MIT)

This repository accompanies our ICML 2025 paper **“Exploring Representations and Interventions in Time Series Foundation Models.”**

## Overview

Time series foundation models (TSFMs) are powerful tools for various applications, but their internal representations and learned concepts are not well understood. In this study, we:

1. **Analyze representation similarity**: Investigate the structure and redundancy of representations across various TSFMs
2. **Perform model pruning**: Leverage redundancy in representations to prune layers and improve efficiency
3. **Identify and localize concepts**: Explore what concepts (periodicity, trends) are learned by these models
4. **Implement concept steering**: Manipulate latent space to influence model behavior

## Repository Structure

```
representations-in-tsfms/
├── efficiency/                # Representation similarity & pruning tools
│   ├── chronos-forecasting/   # Upstream Chronos implementation
│   ├── tsfm_similarity/       # Similarity metrics & experiments
│   ├── produce_similarity_maps.sh
│   ├── produce_and_time_models.sh
│   └── evaluate_chronos_variants.sh
├── steering/                  # Concept discovery & steering
│   ├── configs/               # YAML experiment configs
│   ├── steertool/             # Steering library & CLI
│   ├── run_steering_experiments.sh
│   └── run_separability_analysis.sh
├── environment.yml            # Conda environment specification
└── create_env.sh              # Helper script for env creation
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

### Concept Identification and Steering

Analyze concept separability and steer model behaviour using the provided CLI utilities:

```bash
# move to the steering module
cd steering

# Run separability analysis (creates figures under steering/results)
./run_separability_analysis.sh

# Run steering experiments (latent intervention)
./run_steering_experiments.sh
```

## Citation

```bibtex
@inproceedings{wilinski2025exploring,
  title={Exploring Representations and Interventions in Time Series Foundation Models},
  author={Micha{\l} Wili{\'n}ski and Mononito Goswami and Willa Potosnak and Nina {\.{Z}}ukowska and Artur Dubrawski},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=goVzfYtj58}
}
```

## License

This project is licensed under the MIT License (see the `LICENSE` file for details).
