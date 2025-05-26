#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle


# In[11]:


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


# In[ ]:


with open('reproducible_data.pkl', 'rb') as f:
    data = pickle.load(f)

create_pca_scatter_plot(data['one_activations_plot'], data['other_activations_plot'], 
                        data['one_activations_to_perturb'], data['one_activations_perturbed'], 
                        data['labels'], data['pca'], 'path_to_save_plot_pca.pdf')


# In[ ]:




