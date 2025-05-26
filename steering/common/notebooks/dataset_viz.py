#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


# In[2]:


datasets = [os.path.join('datasets', file) for file in os.listdir(os.path.join(os.getcwd(), 'datasets'))]
datasets = [(file.split('/')[-1].split('.')[0], pd.read_parquet(file)) for file in datasets]


# In[3]:


# Set up Seaborn plot aesthetics with larger font sizes for research papers
sns.set(font_scale=2.0, style="ticks")  # Increased font scale for better readability

def plot_random_samples(X, n_samples=3, save_path=None):
    """
    Plot random samples from time-series data X.
    
    Parameters:
    - X: numpy array or pandas DataFrame of time-series data
    - n_samples: number of random samples to plot
    - save_path: if provided, saves the plot to this path
    """
    
    # Randomly select n_samples from the dataset
    sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    # Create a figure and axes for three subplots in a row (1 row, 3 columns)
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))  # Adjusted figure size for clearer plots

    # Loop through the sample indices and plot each sample on its respective axis
    for i, idx in enumerate(sample_indices):
        y_vals = X[idx].squeeze()
        axes[i].plot(y_vals, linewidth=2)  # Bolden the line
        axes[i].tick_params(axis='both', which='major', labelsize=16)  # Set larger font size for tick labels
        
        # Remove extra ticks from top and right sides for a clean look
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        # Set labels and title with larger font sizes
        axes[i].set_xlabel("Time", fontsize=18)
        axes[i].set_ylabel("Value", fontsize=18) if i == 0 else axes[i].set_ylabel("")
        axes[i].set_title(f"Sample {i+1}", fontsize=20)

    # Adjust layout for better spacing between plots
    plt.tight_layout(pad=2)

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the figure
    plt.show()


def plot_attribute_distributions(dataset, save_path=None):
    """
    Plot distributions of various attributes from the dataset.
    
    Parameters:
    - dataset: pandas DataFrame containing the data
    - save_path: if provided, saves the plot to this path
    """
    # Set the overall aesthetic for the plots with larger font sizes for readability
    sns.set(font_scale=2.0, style="ticks")  # Increased font scale for better readability

    # Define a consistent color for all the plots (e.g., Seaborn default blue)
    color = sns.color_palette()[0]
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    # Create a figure with 2 rows and 2 columns for the 4 attributes
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))  # Adjusted figure size

    # 1. Pattern type distribution (categorical, previously 'seasonality_type')
    sns.countplot(x='seasonality_type', data=dataset, ax=axes[0, 0], color=color, alpha=1.0)
    axes[0, 0].set_title("Pattern Type Distribution", fontsize=22)
    axes[0, 0].set_xlabel("Pattern Type", fontsize=20)
    axes[0, 0].set_ylabel("Count", fontsize=20)

    # 2. Y-intercept distribution (numerical, previously 'trend_intercept')
    sns.histplot(x='trend_intercept', data=dataset, ax=axes[0, 1], color=color, alpha=1.0)
    axes[0, 1].set_title("Y-Intercept Distribution", fontsize=22)
    axes[0, 1].set_xlabel("Y-Intercept", fontsize=20)
    axes[0, 1].set_ylabel("Frequency", fontsize=20)

    # 3. Pattern amplitude distribution (numerical, previously 'seasonality_amplitude')
    sns.histplot(x='seasonality_amplitude', data=dataset, ax=axes[1, 0], color=color, alpha=1.0)
    axes[1, 0].set_title("Pattern Amplitude Distribution", fontsize=22)
    axes[1, 0].set_xlabel("Pattern Amplitude", fontsize=20)
    axes[1, 0].set_ylabel("Frequency", fontsize=20)

    # 4. Pattern period distribution (numerical, previously 'seasonality_period')
    sns.histplot(x='seasonality_period', data=dataset, ax=axes[1, 1], color=color, alpha=1.0)
    axes[1, 1].set_title("Pattern Period Distribution", fontsize=22)
    axes[1, 1].set_xlabel("Pattern Period", fontsize=20)
    axes[1, 1].set_ylabel("Frequency", fontsize=20)

    # Adjust tick labels for better visibility
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Adjust layout for a clean look with enough space between plots
    plt.tight_layout(pad=3.0)

    # Save the figure as PDF with a transparent background (no background)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

    # Show the figure
    plt.show()


# In[ ]:


for idx, (name, dataset) in enumerate(datasets):
    X = dataset['series'].values
    print(f"Dataset: {name}")
    
    # Plot random samples
    plot_random_samples(
        X, n_samples=3, save_path=f"data_viz/timeseries_samples_{name}.pdf"
    )
    
    # Plot attribute distributions
    plot_attribute_distributions(
        dataset, save_path=f"data_viz/appendix_stats_{name}.pdf"
    )


# In[ ]:




