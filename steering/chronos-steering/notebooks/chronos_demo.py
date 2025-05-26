#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from core.separability import compute_and_plot_separability
from core.chronos import get_Chronos, predict_Chronos
from utils import get_sample_from_dataset
import pandas as pd
import torch

model, tokenizer = get_Chronos()
INPUT_SAMPLE = ('datasets/sine_constant.parquet', 0)
sample = get_sample_from_dataset(pd.read_parquet(INPUT_SAMPLE[0]), INPUT_SAMPLE[1]).reshape(1, -1)
print(sample.shape)


# In[ ]:


prediction = predict_Chronos(sample, prediction_length=64)


# In[ ]:


prediction.shape


# In[ ]:


# plot the prediction
import matplotlib.pyplot as plt

plt.plot(prediction.flatten())
plt.show()


# In[ ]:




