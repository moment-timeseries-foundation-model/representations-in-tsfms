#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import os
from core.data_generator import generate_and_save_dataset
from utils import seed_everything


# In[2]:


RANDOM_SEED = 42
seed_everything(RANDOM_SEED)


# ## Example config
# 
# ```yaml
# n_series: 128
# length: 512
# trend_types: ['linear', 'exponential', 'none']
# seasonality_types: ['sine', 'square', 'triangle', 'sawtooth', 'none']
# noise_types: ['gaussian', 'uniform', 'none']
# trend_params:
#   slope: [0.01, 0.1]
#   intercept: [-5, 5]
#   growth_rate: [0.01, 0.1]
# seasonality_params:
#   amplitude: [-5, 5]
#   period: [20, 50]
# noise_params:
#   mean: [-1, 1]
#   stddev: [0.1, 2]
#   low: [-2, 0]
#   high: [0, 2]
# ```

# In[ ]:


configs = [os.path.join('configs', file) for file in os.listdir(os.path.join(os.getcwd(), 'configs'))]
for file in configs:
    print(f"Processing {file}...")
    dataset_filename = file.split('/')[-1].split('.')[0] + '.parquet'
    generate_and_save_dataset(config_file=file, dataset_name=dataset_filename)
    print(f"Done processing {file}...")

