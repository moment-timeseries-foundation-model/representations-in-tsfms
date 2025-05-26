#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
from core.moment import perturb_activations_MOMENT
from core.perturb import add
from core.steering import get_steering_matrix
from utils import get_sample_from_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# In[32]:


METHOD = 'mean'
INPUT_SAMPLE = ('datasets/none_constant.parquet', 0)
ALPHA = 0.0 # 0, 0.25, 0.5, 0.75, 1
PLOT_SAVE_PATH = f"moment_viz/compositional/constant_to_increasing_decreasing_{METHOD}_alpha_{ALPHA}.pdf"


# In[33]:


streering_sine = np.load(f"moment_viz/steering_sine_constant/steering_matrix_{METHOD}.npy")
steering_increase = np.load(f"moment_viz/steering_constant_increasing/steering_matrix_{METHOD}.npy")
steering_decrease = np.load(f"moment_viz/steering_constant_decreasing/steering_matrix_{METHOD}.npy")


# ## Getting samples and steering vectors

# In[34]:


one_sample = get_sample_from_dataset(pd.read_parquet(INPUT_SAMPLE[0]), INPUT_SAMPLE[1])
one_to_other_steering_matrix = (steering_decrease * ALPHA + steering_increase * (1-ALPHA)) / 2


# ## Steer and compare with non-steered

# In[ ]:


non_perturbed_output = perturb_activations_MOMENT(one_sample)
perturbed_output = perturb_activations_MOMENT(one_sample, perturbation_fn=add, perturbation_payload=2*one_to_other_steering_matrix)
non_perturbed_output, perturbed_output = non_perturbed_output.flatten(), perturbed_output.flatten()


# In[ ]:


def plot_imputed_signals_with_smoothing(imputed_normal, imputed_perturbed, window_size=10, save_path=None):
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




