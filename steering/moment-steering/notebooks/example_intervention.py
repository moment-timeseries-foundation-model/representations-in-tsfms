#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_dataset
from core.moment import perturb_activations_MOMENT


# In[ ]:


single_sample = load_dataset("datasets/sine_increasing.parquet", type="torch")[0:1,:,:]
print(single_sample.shape)
model_output = perturb_activations_MOMENT(single_sample)


# In[4]:


sns.set(font_scale=2.0, style="ticks")  # Keep the larger font scale for research papers

def plot_single_sample(sample, save_path=None):
    """
    Plot a single time-series sample.
    
    Parameters:
    - sample: A single time-series data array or list.
    - save_path: if provided, saves the plot to this path
    """
    # Create a figure for the single plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(6, 4))  # Adjusted figure size for a single plot
    # Plot the sample
    y_vals = sample.squeeze()
    ax.plot(y_vals, linewidth=3)  # Bolden the line
    ax.tick_params(axis='both', which='major', labelsize=16)  # Set larger font size for tick labels
    
    # Remove extra ticks from top and right sides for a clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set labels and title with larger font sizes
    ax.set_xlabel("Time", fontsize=18)
    ax.set_ylabel("Value", fontsize=18)
    ax.set_title("Sample Plot", fontsize=20)

    # Adjust layout
    plt.tight_layout(pad=2)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the figure
    plt.show()


# In[ ]:


plot_single_sample(model_output)


# In[ ]:




