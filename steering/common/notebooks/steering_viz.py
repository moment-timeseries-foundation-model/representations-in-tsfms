#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from src.moment import perturb_activations_MOMENT
from src.perturb import add
from src.steering import get_steering_matrix
from src.utils import get_sample_from_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# In[8]:


METHOD = 'mean'
LAYER_TO_PLOT = 18
ONE_ACTIVATIONS = 'activations_moment/none_constant_activations.npy'
OTHER_ACTIVATIONS = 'activations_moment/sine_constant_activations.npy'
INPUT_SAMPLE = ('datasets/none_constant.parquet', 0)
PLOT_SAVE_PATH = f"moment_viz/steering_sine_constant/constant_to_sine_{METHOD}.pdf"


# ## Getting samples and steering vectors

# In[9]:


one_activations = np.load(ONE_ACTIVATIONS) # (layer, batch, patch, features)
other_activations = np.load(OTHER_ACTIVATIONS)
one_sample = get_sample_from_dataset(pd.read_parquet(INPUT_SAMPLE[0]), INPUT_SAMPLE[1])
one_to_other_steering_matrix = get_steering_matrix(one_activations, other_activations, method=METHOD)


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pickle

def create_pca_scatter_plot(one_activations_plot, other_activations_plot, 
                            one_activations_to_perturb, one_activations_perturbed, 
                            labels, pca, savepath):
    """
    Creates and saves a PCA scatter plot for activations and perturbed activations.
    
    Parameters:
    - one_activations_plot: np.array
    - other_activations_plot: np.array
    - one_activations_to_perturb: np.array
    - one_activations_perturbed: np.array
    - labels: np.array
    - pca: PCA object used for transformation
    - savepath: string where the plot will be saved
    """
    
    # Perform PCA transformation for the datasets
    dataset_with_labels = np.concatenate([one_activations_plot, other_activations_plot], axis=0)
    pca_result = pca.fit_transform(dataset_with_labels)
    pca_df = pd.DataFrame(pca_result, columns=['pca1', 'pca2'])
    pca_df['labels'] = labels

    one_activations_to_perturb_pca = pca.transform(one_activations_to_perturb)
    one_activations_perturbed_pca = pca.transform(one_activations_perturbed)

    # Set the style for the plot
    sns.set(font_scale=2.0, style="ticks")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rc('font', family='serif')

    plt.figure(figsize=(15, 10))

    # Create scatterplot
    sns.scatterplot(x='pca1', y='pca2', hue='labels', data=pca_df, palette=['blue', 'red'], alpha=0.6, s=150, edgecolor='k')

    for i in range(one_activations_to_perturb_pca.shape[0]):
        plt.scatter(one_activations_to_perturb_pca[i, 0], one_activations_to_perturb_pca[i, 1], 
                    c='green', label='Original to Perturbed' if i == 0 else '', alpha=0.9, s=200, edgecolor='k')
        plt.scatter(one_activations_perturbed_pca[i, 0], one_activations_perturbed_pca[i, 1], 
                    c='orange', label='Perturbed' if i == 0 else '', alpha=0.9, s=200, edgecolor='k')

        # Arrow showing the direction of the perturbation
        direction = one_activations_perturbed_pca[i] - one_activations_to_perturb_pca[i]
        offset = 4  
        length = np.linalg.norm(direction) - offset
        direction_normalized = direction / np.linalg.norm(direction) * length
        plt.arrow(one_activations_to_perturb_pca[i, 0], one_activations_to_perturb_pca[i, 1],
                  direction_normalized[0], direction_normalized[1],
                  color='black', alpha=0.7, head_width=2, head_length=2, linewidth=3)

    plt.xlabel('Principal Component 1', fontsize=26, family='serif')
    plt.ylabel('Principal Component 2', fontsize=26, family='serif')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = {'Class 0': handles[0], 'Class 1': handles[1]}
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='best', fontsize=26, frameon=True)

    plt.tick_params(axis='both', which='major', labelsize=26)

    # Save the plot
    plt.savefig(f"{savepath}", bbox_inches='tight')
    plt.show()

# To make this reproducible, save all necessary values
def save_reproducible_data(filename, one_activations_plot, other_activations_plot, 
                           one_activations_to_perturb, one_activations_perturbed, 
                           labels, pca):
    """
    Saves all necessary values to reproduce the PCA plot.
    
    Parameters:
    - filename: string, path where to save the file
    - one_activations_plot, other_activations_plot, one_activations_to_perturb, one_activations_perturbed: np.array
    - labels: np.array
    - pca: PCA object
    """
    with open(filename, 'wb') as f:
        pickle.dump({
            'one_activations_plot': one_activations_plot,
            'other_activations_plot': other_activations_plot,
            'one_activations_to_perturb': one_activations_to_perturb,
            'one_activations_perturbed': one_activations_perturbed,
            'labels': labels,
            'pca': pca
        }, f)


# Random example of how to call these functions assuming one_activations, other_activations are available
one_activations_plot = one_activations[LAYER_TO_PLOT, :30, :, :]
other_activations_plot = other_activations[LAYER_TO_PLOT, :30, :, :]

# Selecting 5 random activations for perturbation
random_indices = np.random.choice(one_activations_plot.shape[0], 5, replace=False)
one_activations_to_perturb = one_activations_plot[random_indices]
steering_matrix_for_layer = one_to_other_steering_matrix[LAYER_TO_PLOT]
one_activations_perturbed = one_activations_to_perturb.copy() + steering_matrix_for_layer

# Averaging activations over patches
one_activations_plot = np.mean(one_activations_plot, axis=1)
other_activations_plot = np.mean(other_activations_plot, axis=1)
one_activations_to_perturb = np.mean(one_activations_to_perturb, axis=1)
one_activations_perturbed = np.mean(one_activations_perturbed, axis=1)

# Reshape for PCA
one_activations_plot = one_activations_plot.reshape(-1, one_activations_plot.shape[-1])
other_activations_plot = other_activations_plot.reshape(-1, other_activations_plot.shape[-1])

# Concatenate datasets
dataset_with_labels = np.concatenate([one_activations_plot, other_activations_plot], axis=0)
labels = np.concatenate([np.zeros(one_activations_plot.shape[0]), np.ones(other_activations_plot.shape[0])])

# Fit PCA
pca = PCA(n_components=2)

# Save all the values needed to reproduce the plot
save_reproducible_data('reproducible_data.pkl', one_activations_plot, other_activations_plot, 
                       one_activations_to_perturb, one_activations_perturbed, labels, pca)

# Create the PCA plot
create_pca_scatter_plot(one_activations_plot, other_activations_plot, one_activations_to_perturb, 
                        one_activations_perturbed, labels, pca, PLOT_SAVE_PATH)


# In[ ]:


with open('reproducible_data.pkl', 'rb') as f:
    data = pickle.load(f)

create_pca_scatter_plot(data['one_activations_plot'], data['other_activations_plot'], 
                        data['one_activations_to_perturb'], data['one_activations_perturbed'], 
                        data['labels'], data['pca'], 'path_to_save_plot_pca.pdf')


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




