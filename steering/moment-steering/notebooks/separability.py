#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from core.separability import compute_and_plot_separability
import numpy as np
sine_constant_activations = np.load('activations/sine_amp_high_activations.npy')[:, :128, :, :]
none_constant_activations = np.load('activations/sine_amp_low_activations.npy')[:, :128, :, :]
# high/low sine/constant increasing/decreasing
compute_and_plot_separability(sine_constant_activations, none_constant_activations, prefix='moment_viz/sine_amplitude/')


# In[ ]:




