#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from src.moment import perturb_activations_MOMENT, get_activations_MOMENT
from src.perturb import add
from src.steering import get_steering_matrix
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.io.arff import loadarff 


# In[2]:


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# In[ ]:


def get_ecg_dataset():
    train = loadarff('ECG5000/ECG5000_TRAIN.arff')
    test = loadarff('ECG5000/ECG5000_TEST.arff')
    train_df = pd.DataFrame(train[0])
    test_df = pd.DataFrame(test[0])

    # detach the data from the labels
    train_data = train_df.iloc[:, :-1].values
    train_labels = train_df.iloc[:, -1].values
    test_data = test_df.iloc[:, :-1].values
    test_labels = test_df.iloc[:, -1].values

    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])

    train_labels = np.array([0 if x == b'1' else 1 for x in train_labels])
    test_labels = np.array([0 if x == b'1' else 1 for x in test_labels])
    return train_data, train_labels, test_data, test_labels

train_data, train_labels, test_data, test_labels = get_ecg_dataset()
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)


# In[4]:


from src.moment import train_classifier_MOMENT, perform_classification_MOMENT, get_MOMENT
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# In[ ]:


svm_classifier = train_classifier_MOMENT(train_data, train_labels, device=DEVICE)
moment = get_MOMENT("embedding", device=DEVICE)
y_hat = perform_classification_MOMENT(test_data, svm_classifier, moment)


# In[ ]:


print("Accuracy: ", accuracy_score(test_labels, y_hat))
print("F1 Score: ", f1_score(test_labels, y_hat))
print("AUC: ", roc_auc_score(test_labels, y_hat))


# # Steering

# In[7]:


METHOD = 'median'
PLOT_SAVE_PATH = f"real_world_abnormal_to_normal_{METHOD}.pdf"
LAYER_TO_PLOT = 17


# In[ ]:


test_data[test_labels == 0].shape


# In[ ]:


normal_activations = get_activations_MOMENT(test_data[test_labels == 0], device=DEVICE)
abnormal_activations = get_activations_MOMENT(test_data[test_labels == 1], device=DEVICE)


# In[ ]:


print(normal_activations.shape, abnormal_activations.shape)


# In[11]:


steering_ab_to_normal = get_steering_matrix(abnormal_activations, normal_activations, method=METHOD)


# In[ ]:


one_activations = normal_activations
other_activations = abnormal_activations
one_sample = one_activations[0]
one_to_other_steering_matrix = -steering_ab_to_normal
# Assume one_activations, other_activations, and one_to_other_steering_matrix are already loaded
one_activations_plot = one_activations[LAYER_TO_PLOT, :, :, :]
other_activations_plot = other_activations[LAYER_TO_PLOT, :, :, :]

# Select 5 random activations from the first dataset for perturbation
random_indices = np.random.choice(one_activations_plot.shape[0], 5, replace=False)
one_activations_to_perturb = one_activations_plot[random_indices]
steering_matrix_for_layer = one_to_other_steering_matrix[LAYER_TO_PLOT]
one_activations_perturbed = one_activations_to_perturb.copy() + steering_matrix_for_layer

# Average the activations over the patches
one_activations_plot = np.mean(one_activations_plot, axis=1)
other_activations_plot = np.mean(other_activations_plot, axis=1)
one_activations_to_perturb = np.mean(one_activations_to_perturb, axis=1)
one_activations_perturbed = np.mean(one_activations_perturbed, axis=1)

# Perform PCA on the activations
one_activations_plot = one_activations_plot.reshape(-1, one_activations_plot.shape[-1])
other_activations_plot = other_activations_plot.reshape(-1, other_activations_plot.shape[-1])

dataset_with_labels = np.concatenate([one_activations_plot, other_activations_plot], axis=0)
labels = np.concatenate([np.zeros(one_activations_plot.shape[0]), np.ones(other_activations_plot.shape[0])])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(dataset_with_labels)
pca_df = pd.DataFrame(pca_result, columns=['pca1', 'pca2'])
pca_df['labels'] = labels

# Apply PCA transformation on the original and perturbed activations
one_activations_to_perturb_pca = pca.transform(one_activations_to_perturb)
one_activations_perturbed_pca = pca.transform(one_activations_perturbed)

# Set font and style similar to the imputed signals plot
sns.set(font_scale=2.0, style="ticks")
plt.style.use('seaborn-v0_8-whitegrid')
plt.rc('font', family='serif')

# Plotting
plt.figure(figsize=(15, 10))

# Plot the original activations
sns.scatterplot(x='pca1', y='pca2', hue='labels', data=pca_df, palette=['blue', 'red'], alpha=0.6, s=150, edgecolor='k')

# Plot the perturbed activations and arrows
for i in range(one_activations_to_perturb_pca.shape[0]):
    # Ensure the label 'Original to Perturbed' appears only once
    plt.scatter(one_activations_to_perturb_pca[i, 0], one_activations_to_perturb_pca[i, 1], c='green', label='Original to Perturbed' if i == 0 else '', alpha=0.9, s=200, edgecolor='k')
    plt.scatter(one_activations_perturbed_pca[i, 0], one_activations_perturbed_pca[i, 1], c='orange', label='Perturbed' if i == 0 else '', alpha=0.9, s=200, edgecolor='k')

    # Draw arrows indicating the shift from original to perturbed
    direction = one_activations_perturbed_pca[i] - one_activations_to_perturb_pca[i]
    offset = 1.2  # Adjust to stop the arrow just before the perturbed point
    length = np.linalg.norm(direction) - offset
    direction_normalized = direction / np.linalg.norm(direction) * length
    plt.arrow(one_activations_to_perturb_pca[i, 0], one_activations_to_perturb_pca[i, 1],
              direction_normalized[0], direction_normalized[1],
              color='black', alpha=0.7, head_width=0.5, head_length=0.5, linewidth=3)

# Add labels, legend, and title with adjusted fonts
plt.xlabel('Principal Component 1', fontsize=26, family='serif')
plt.ylabel('Principal Component 2', fontsize=26, family='serif')

# Modify legend to remove redundant entries
handles, labels = plt.gca().get_legend_handles_labels()

unique_labels = {'Class 0': handles[0], 'Class 1': handles[1]}
plt.legend(unique_labels.values(), unique_labels.keys(), loc='best', fontsize=26, frameon=True)

# Adjust tick parameters for better visibility
plt.tick_params(axis='both', which='major', labelsize=26)

# Save the plot as a high-resolution PDF # insert pca just before .pdf in the string
savepath = PLOT_SAVE_PATH[:-4] + '_pca.pdf'
plt.savefig(f"{savepath}", bbox_inches='tight')

# Show plot
plt.show()


# ## Steer and compare with non-steered

# In[ ]:


steering_ab_to_normal.shape


# In[ ]:


single_abnormal = test_data[test_labels == 1][5].reshape(1, 1, -1)
print(single_abnormal.shape)


# In[ ]:


def simple_perturbation(sample, steering_matrix, space='reconstruction'):
    original_length = sample.shape[-1]
    if steering_matrix.shape[1] != 64 and False:
        steering_matrix = np.concatenate([steering_matrix, np.zeros((24, 47, 1024))], axis=1)
    steering_matrix = torch.tensor(steering_matrix, device=DEVICE)
    if sample.shape[2] != 512 and False:
        sample = np.concatenate([sample, np.zeros((1, 1, 512-140))], axis=-1)
    non_perturbed_output = perturb_activations_MOMENT(torch.tensor(sample, dtype=torch.float32), space=space, device=DEVICE)
    perturbed_output = perturb_activations_MOMENT(torch.tensor(sample, dtype=torch.float32), perturbation_fn=add, perturbation_payload=steering_matrix, space=space, device=DEVICE)
    non_perturbed_output = non_perturbed_output.flatten()
    perturbed_output = perturbed_output.flatten()
    if space == 'reconstruction':
        return non_perturbed_output[:original_length], perturbed_output[:original_length]
    return non_perturbed_output, perturbed_output
    
non_perturbed_output, perturbed_output = simple_perturbation(single_abnormal, steering_ab_to_normal, space='reconstruction') 


# In[16]:


non_perturbed_output, perturbed_output = simple_perturbation(single_abnormal, steering_ab_to_normal, space='reconstruction') 


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


multiple_abnormal_samples = test_data[test_labels == 1]
abnormal_embedded = []
perturbed_embedded = []
for i in range(15):
    non_perturbed_output, perturbed_output = simple_perturbation(multiple_abnormal_samples[i].reshape(1, 1, -1), 0.6*steering_ab_to_normal, space='embedding')
    abnormal_embedded.append(non_perturbed_output)
    perturbed_embedded.append(perturbed_output)

abnormal_embedded = np.array(abnormal_embedded)
perturbed_embedded = np.array(perturbed_embedded)


# In[ ]:


embedded_test = moment.embed(x_enc=torch.tensor(multiple_abnormal_samples[0].reshape(1,1,-1), device=DEVICE,dtype=torch.float32)).embeddings.detach().cpu().numpy()
embedded_test


# In[ ]:


y_hats_abnormal = svm_classifier.predict(abnormal_embedded)
y_hats_perturbed = svm_classifier.predict(perturbed_embedded)

print("abNormal Accuracy: ", accuracy_score(np.ones(15), y_hats_abnormal))
print("Perturbed Accuracy: ", accuracy_score(np.ones(15), y_hats_perturbed))


# In[21]:


# check if all labels have been swapped
swaps = np.sum(y_hats_abnormal != y_hats_perturbed)


# In[ ]:


print(y_hats_abnormal)
print(y_hats_perturbed)


# In[ ]:




