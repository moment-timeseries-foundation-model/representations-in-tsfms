#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
from core.chronos import perturb_activations_Chronos
from core.perturb import add
from core.steering import get_steering_matrix
from utils import get_sample_from_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# In[52]:


METHOD = 'mean'
LAYER_TO_PLOT = 23
CONSTANT_ACTIVATIONS = 'activations_chronos/none_constant_activations.npy'
INCREASING_ACTIVATIONS = 'activations_chronos/none_increasing_activations.npy'
SINE_ACTIVATIONS = 'activations_chronos/sine_constant_activations.npy'
INPUT_SAMPLE = ('datasets/none_constant.parquet', 0)
ALPHA = 0.2
CONVEX = 0.0
PLOT_SAVE_PATH = f"chronos_viz/compositional/sine_vs_increasing_{METHOD}_alpha_{CONVEX}.pdf"


# ## Getting samples and steering vectors

# In[ ]:


constant_activations = np.load(CONSTANT_ACTIVATIONS)
increasing_activations = np.load(INCREASING_ACTIVATIONS)
sine_activations = np.load(SINE_ACTIVATIONS)
constant_sample = get_sample_from_dataset(pd.read_parquet(INPUT_SAMPLE[0]), INPUT_SAMPLE[1])

constant_to_increasing_steering_matrix = get_steering_matrix(constant_activations, increasing_activations, method=METHOD)
constant_to_sine_steering_matrix = get_steering_matrix(constant_activations, sine_activations, method=METHOD)


# In[53]:


steering_matrix = CONVEX * constant_to_increasing_steering_matrix + (1 - CONVEX) * constant_to_sine_steering_matrix


# ## Steer and compare with non-steered

# In[ ]:


non_perturbed_output = perturb_activations_Chronos(constant_sample)
perturbed_output = perturb_activations_Chronos(constant_sample, perturbation_fn=add, perturbation_payload=ALPHA*steering_matrix)
non_perturbed_output, perturbed_output = non_perturbed_output.flatten(), perturbed_output.flatten()


# In[ ]:


def plot_imputed_signals_with_smoothing(imputed_normal, imputed_perturbed, window_size=5, save_path=None):
    """
    Plot imputed signals (normal and perturbed) for time series data with smoothing.
    Show non-smoothed series in pale colors and smoothed series in bold colors.
    
    Parameters:
    - imputed_normal: The normal imputed signal (numpy array).
    - imputed_perturbed: The perturbed imputed signal (numpy array).
    - window_size: The window size for the moving average smoothing.
    - save_path: If provided, saves the plot to this path as PDF.
    """
    palette = sns.color_palette()
    # Convert arrays to pandas Series and apply rolling window (moving average)
    imputed_normal_smoothed = pd.Series(imputed_normal).rolling(window=window_size, center=True).mean()
    imputed_perturbed_smoothed = pd.Series(imputed_perturbed).rolling(window=window_size, center=True).mean()

    # Set style and fonts for the plot
    sns.set(font_scale=2.0, style="ticks")
    # use seaborn v08 white grid
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rc('font', family='serif')  # Ensure serif font is used for the plot

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot non-smoothed series in pale colors (using same colors with alpha for transparency)
    ax.plot(imputed_perturbed, label='Perturbed', color=palette[0], alpha=0.6, linewidth=2)
    ax.plot(imputed_normal, label='Non-Perturbed', color=palette[1], alpha=0.6, linewidth=2)

    # Plot smoothed series in bold colors (using same colors with higher opacity)
    ax.plot(imputed_perturbed_smoothed, label='Perturbed (Smoothed)', color=palette[0], linewidth=3)
    ax.plot(imputed_normal_smoothed, label='Non-Perturbed (Smoothed)', color=palette[1], linewidth=3)

    # Title and labels with explicit font size
    #ax.set_title("Steering Effect on Model Output", fontsize=20)
    ax.set_xlabel("Timestep", fontsize=20)

    # Add legend
    ax.legend(loc='best', fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    # remove smoothed word from the perturbed (smoothed) and non-perturbed (smoothed) labels
    # remvoe at all the perturbed and non-perturbed labels
    labels = [label.replace(' (Smoothed)', '') for label in labels]
    # remove the pale lines from the legend
    handles = [handles[2], handles[3]]
    ax.legend(handles, labels, loc='best', fontsize=20, frameon=True)
    
    # Adjust layout for clarity
    plt.tight_layout(pad=2)

    # Save as PDF if path is provided
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

# Example usage
plot_imputed_signals_with_smoothing(non_perturbed_output, perturbed_output, save_path=PLOT_SAVE_PATH)


# In[ ]:





# In[ ]:





# In[ ]:




