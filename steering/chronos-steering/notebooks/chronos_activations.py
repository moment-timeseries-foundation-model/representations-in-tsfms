#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from core.separability import compute_and_plot_separability
from core.chronos import get_Chronos, predict_Chronos, get_activations_Chronos
from utils import get_sample_from_dataset, load_dataset
import pandas as pd
import torch

dataset = load_dataset('datasets/sine_constant.parquet', type="numpy").squeeze(1)


# In[ ]:


dataset.shape
dataset = dataset[0].reshape(1, -1)
dataset.shape


# In[ ]:


activations_encoder, activations_decoder = get_activations_Chronos(dataset)


# In[ ]:


print(activations_encoder.shape)


# In[ ]:


print(activations_decoder.shape)


# In[ ]:


# plot the prediction
import matplotlib.pyplot as plt

plt.plot(prediction.flatten())
plt.show()


# In[ ]:




