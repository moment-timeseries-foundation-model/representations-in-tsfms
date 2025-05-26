#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from core.separability import compute_and_plot_separability
import numpy as np
sine_constant_activations = np.load('activations_chronos/sine_constant_activations.npy')
none_constant_activations = np.load('activations_chronos/none_constant_activations.npy')
# high/low sine/constant increasing/decreasing
compute_and_plot_separability(sine_constant_activations, none_constant_activations, prefix='chronos_viz/sine_constant_vs_none_constant/')


# In[ ]:


none_constant_activations.shape


# In[ ]:




