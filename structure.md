# Project Structure

## Steering Directory Structure

The `steering` directory has been reorganized with the following structure:

```
steering/
├── common/                     # Shared resources between projects
│   ├── configs/                # Configuration files
│   ├── datasets/               # Datasets including ECG5000
│   ├── notebooks/              # General notebooks
│   │   ├── steering.ipynb
│   │   ├── steer_minimal.ipynb
│   │   ├── steering_viz.ipynb
│   │   ├── real_world_normal_to_abnormal.ipynb
│   │   ├── real_world_ab_to_normal.ipynb
│   │   ├── dataset_viz.ipynb
│   │   └── dataset_generation.ipynb
│   └── src/                    # Shared source code
│
├── chronos-steering/           # Chronos-specific steering project
│   ├── notebooks/              # Chronos-specific notebooks
│   │   ├── steering_viz_chronos.ipynb
│   │   ├── separability_chronos.ipynb
│   │   ├── compositional_chronos.ipynb
│   │   ├── chronos_demo.ipynb
│   │   └── chronos_activations.ipynb
│   ├── scripts/                # Chronos-specific scripts
│   │   └── generate_activations_chronos.sh
│   └── viz/                    # Chronos visualization data
│
└── moment-steering/            # Moment-specific steering project
    ├── notebooks/              # Moment-specific notebooks
    │   ├── real_world.ipynb
    │   ├── separability.ipynb
    │   ├── example_intervention.ipynb
    │   └── compositional_steering.ipynb
    ├── scripts/                # Moment-specific scripts
    │   └── generate_activations_moment.sh
    └── viz/                    # Moment visualization data
```

This reorganization follows a similar pattern to the `efficiency` directory, making it easier to:
1. Find model-specific files (chronos vs. moment)
2. Share common resources between projects
3. Separate different types of files (notebooks, scripts, visualizations) 


I've just converted all the notebooks to .py scripts. 
A lot of code is repeated between chronos and moment and some other places. I want everything written as a simple 