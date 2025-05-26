import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
from itertools import product
from tqdm import tqdm


def compute_linear_separability(
    layer, patch, activations_class_one, activations_class_other, no_samples
):
    """
    Computes the Fisher's Linear Discriminant Ratio for a given layer and patch.

    Parameters:
    layer: int, current layer index
    patch: int, current patch index
    activations_class_one: numpy array, activations for class one (sine_constant)
    activations_class_other: numpy array, activations for class other (none_constant)
    no_samples: int, number of samples

    Returns:
    tuple: layer, patch, and Fisher's score
    """
    # Combine activations and labels
    activations = np.concatenate(
        (activations_class_one, activations_class_other), axis=0
    )
    labels = np.concatenate((np.ones(no_samples), np.zeros(no_samples)))

    # Fit LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(activations, labels)

    # Project activations onto LDA direction
    projections = lda.transform(activations)
    projections_class_one = projections[:no_samples]
    projections_class_other = projections[no_samples:]

    # Compute means and variances of projections
    mean1 = projections_class_one.mean()
    mean2 = projections_class_other.mean()
    var1 = projections_class_one.var()
    var2 = projections_class_other.var()

    # Compute Fisher's Discriminant Ratio
    fisher_score = ((mean1 - mean2) ** 2) / (var1 + var2)

    return layer, patch, fisher_score


def plot_linear_separability(
    linear_separability,
    linear_separability_mean,
    no_layers,
    title="Linear Separability Heatmap",
    output_file="linear_separability_heatmap.pdf",
):
    """
    Plots a heatmap of linear separability and an overlay of mean separability across layers.

    Parameters:
    linear_separability: 2D numpy array for heatmap data (shape: patches x layers)
    linear_separability_mean: 1D numpy array of mean separability for each layer
    no_layers: int, number of layers (x-axis length)
    title: str, title for the plot
    output_file: str, the file name to save the plot
    """

    # Set up Seaborn plot aesthetics
    sns.set(font_scale=2.0, style="ticks")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    # Define plot size
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Adjust the space around the heatmap to fit the color bar better
    fig.subplots_adjust(right=0.85, top=0.92, bottom=0.1, left=0.1)

    # Plot heatmap for linear separability, transposed so that layers are on x-axis
    cbar_ax = fig.add_axes([1.07, 0.18, 0.03, 0.65])
    sns.heatmap(
        linear_separability.T, cmap="viridis", ax=ax1, cbar=True, cbar_ax=cbar_ax
    )

    # X-axis and Y-axis labels with increased font sizes and label padding
    ax1.set_xlabel("Layer", fontsize=22, labelpad=20)
    ax1.set_ylabel("Patch", fontsize=22, labelpad=20)

    # Title with increased font size and padding
    ax1.set_title(title, fontsize=24, pad=30)

    # Create a secondary y-axis for the mean linear separability across layers
    ax2 = ax1.twinx()

    # Plot the mean linear separability across layers on the secondary y-axis
    ax2.plot(
        np.arange(no_layers),
        linear_separability_mean,
        color="red",
        linewidth=2,
        label="Mean Separability",
    )

    # Align x-axis of the line plot with the layers on the heatmap
    ax2.set_xlim(0, no_layers - 1)
    ax2.set_ylabel("Separability", fontsize=22, labelpad=20)

    # Improve spacing and layout with more padding for texts and elements
    plt.tight_layout()

    # Save the combined plot
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    print(f"Plot saved as {output_file}")

def visualize_embeddings_pca(
    one_activations,
    other_activations,
    coordinates,
    title="Layer Embeddings - PCA with Shifted Samples",
    output_file="embedding_visualization.pdf",
):
    """
    Visualize the embeddings in a selected layer and patch after applying PCA 
    and highlight separability between sine and none samples.

    Parameters:
    sine_constant_activations: numpy array, activations for sine_constant input
    none_constant_activations: numpy array, activations for none_constant input
    layer_to_visualize: int, the layer index to visualize
    patch: int, patch index to visualize
    title: str, the title for the plot
    output_file: str, the file name to save the plot
    """
    
    if isinstance(coordinates, tuple):
        layer_to_visualize, patch = coordinates
        one_patch_embeddings = one_activations[layer_to_visualize, :, patch, :]
        other_patch_embeddings = other_activations[layer_to_visualize, :, patch, :]
        
    else:
        layer_to_visualize = coordinates
        one_patch_embeddings = np.mean(one_activations[layer_to_visualize, :, :, :], axis=1)
        other_patch_embeddings = np.mean(other_activations[layer_to_visualize, :, :, :], axis=1)

    # Apply PCA on the activations for this patch separately
    pca = PCA(n_components=2)
    one_reduced = pca.fit_transform(one_patch_embeddings)
    other_reduced = pca.transform(other_patch_embeddings)

    # Set up Seaborn plot aesthetics to match the main plot
    sns.set(font_scale=2.0, style="ticks")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 10))

    # Plot embeddings for the selected patch
    ax1.scatter(
        one_reduced[:, 0],
        one_reduced[:, 1],
        c="blue",
        label="Class 0",
        alpha=0.6,
    )
    ax1.scatter(
        other_reduced[:, 0],
        other_reduced[:, 1],
        c="red",
        label="Class 1",
        alpha=0.6,
    )

    # Add labels, legend, and title with appropriate styling
    ax1.set_title(title, fontsize=24, pad=30)
    ax1.set_xlabel("Principal Component 1", fontsize=22, labelpad=20)
    ax1.set_ylabel("Principal Component 2", fontsize=22, labelpad=20)
    ax1.legend(loc="best", fontsize=18)
    ax1.grid(True)

    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    print(f"Embedding visualization saved as {output_file}")
    
def visualize_embeddings_lda(
    one_activations,
    other_activations,
    coordinates,
    title="Layer Embeddings - LDA with Shifted Samples",
    output_file="embedding_visualization_lda_1d.pdf",
):
    """
    Visualize the embeddings in a selected layer and patch after applying LDA 
    and highlight separability between two classes (Class 0 and Class 1).
    This version reduces the embeddings to one dimension.

    Parameters:
    one_activations: numpy array, activations for Class 0 input
    other_activations: numpy array, activations for Class 1 input
    coordinates: tuple or int, layer and patch indices or just layer index for visualization
    title: str, the title for the plot
    output_file: str, the file name to save the plot
    """
    
    # Extract embeddings based on the provided coordinates
    if isinstance(coordinates, tuple):
        layer_to_visualize, patch = coordinates
        one_patch_embeddings = one_activations[layer_to_visualize, :, patch, :]
        other_patch_embeddings = other_activations[layer_to_visualize, :, patch, :]
        
    else:
        layer_to_visualize = coordinates
        one_patch_embeddings = np.mean(one_activations[layer_to_visualize, :, :, :], axis=1)
        other_patch_embeddings = np.mean(other_activations[layer_to_visualize, :, :, :], axis=1)

    # Create labels for LDA
    labels = np.concatenate([np.zeros(one_patch_embeddings.shape[0]), np.ones(other_patch_embeddings.shape[0])])
    
    # Stack embeddings from both classes
    combined_embeddings = np.vstack([one_patch_embeddings, other_patch_embeddings])

    # Apply LDA to reduce dimensionality to 1 component
    lda = LinearDiscriminantAnalysis(n_components=1)
    reduced_embeddings = lda.fit_transform(combined_embeddings, labels)

    # Split back the reduced embeddings into class 0 and class 1
    one_reduced = reduced_embeddings[labels == 0]
    other_reduced = reduced_embeddings[labels == 1]

    # Set up Seaborn plot aesthetics to match the main plot
    sns.set(font_scale=2.0, style="ticks")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot embeddings for the selected patch in 1D
    ax1.scatter(
        one_reduced[:, 0],
        np.zeros(one_reduced.shape[0]),  # Use zeros for the y-axis to plot in 1D
        c="blue",
        label="Class 0",
        alpha=0.6,
    )
    ax1.scatter(
        other_reduced[:, 0],
        np.zeros(other_reduced.shape[0]),  # Use zeros for the y-axis to plot in 1D
        c="red",
        label="Class 1",
        alpha=0.6,
    )

    # Add labels, legend, and title with appropriate styling
    ax1.set_title(title, fontsize=24, pad=30)
    ax1.set_xlabel("LDA Component 1", fontsize=22, labelpad=20)
    ax1.set_yticks([])  # Remove y-axis ticks since we are plotting in 1D
    ax1.legend(loc="best", fontsize=18)
    ax1.grid(True)

    # Save the visualization
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    print(f"Embedding visualization saved as {output_file}")


def compute_and_plot_separability(
    one_activations,
    other_activations,
    prefix="default_model/",
):
    """
    Main function to compute linear separability and generate the plot, and
    visualize PCA for highest and lowest separability.

    Parameters:
    sine_constant_activations: numpy array, activations for sine_constant input
    none_constant_activations: numpy array, activations for none_constant input
    output_file: str, name of the output image file for the separability heatmap
    embedding_high_output_file: str, name of the output image file for the PCA visualization of highest separability
    embedding_low_output_file: str, name of the output image file for the PCA visualization of lowest separability
    """

    no_layers, no_samples, no_patches, no_features = one_activations.shape

    # Initialize linear separability matrix
    linear_separability = np.zeros((no_layers, no_patches))

    # Generate all combinations of layers and patches
    layer_patch_combinations = list(product(range(no_layers), range(no_patches)))

    # Use joblib to parallelize over layers and patches
    results = Parallel(n_jobs=-1)(
        delayed(compute_linear_separability)(
            layer,
            patch,
            one_activations[layer, :, patch, :],
            other_activations[layer, :, patch, :],
            no_samples,
        )
        for layer, patch in tqdm(layer_patch_combinations, desc="Processing layers and patches")
    )

    # Update the linear separability matrix with results
    for layer_idx, patch_idx, score in results:
        linear_separability[layer_idx, patch_idx] = score

    # Perform min-max scaling
    linear_separability = (linear_separability - linear_separability.min()) / (
        linear_separability.max() - linear_separability.min()
    )

    # Calculate the mean over patches for each layer
    sine_average = np.mean(one_activations, axis=2, keepdims=True)
    none_average = np.mean(other_activations, axis=2, keepdims=True)

    # Initialize linear separability matrix for averaged activations
    linear_separability_mean = np.zeros((no_layers))

    # Use joblib to parallelize over layers
    results_mean = Parallel(n_jobs=-1)(
        delayed(compute_linear_separability)(
            layer,
            0,
            sine_average[layer, :, 0, :],
            none_average[layer, :, 0, :],
            no_samples,
        )
        for layer in range(no_layers)
    )

    # Update the linear separability mean array with results
    for layer_idx, _, score in results_mean:
        linear_separability_mean[layer_idx] = score

    # Perform min-max scaling for the mean separability
    linear_separability_mean = (
        linear_separability_mean - linear_separability_mean.min()
    ) / (linear_separability_mean.max() - linear_separability_mean.min())

    # Plot the results
    plot_linear_separability(
        linear_separability,
        linear_separability_mean,
        no_layers,
        output_file=prefix + "linear_separability_heatmap.pdf",
    )

    # Identify patches with highest and lowest separability
    patch_highest = np.unravel_index(
        np.argmax(linear_separability, axis=None), linear_separability.shape
    )
    patch_lowest = np.unravel_index(
        np.argmin(linear_separability, axis=None), linear_separability.shape
    )
    
    layer_mean_higest = np.argmax(linear_separability_mean)
    layer_mean_lowest = np.argmin(linear_separability_mean)

    # Visualize the embeddings using PCA for the highest separability patch
    visualize_embeddings_pca(
        one_activations,
        other_activations,
        patch_highest,
        title=f"Layer {patch_highest[0]} - Patch {patch_highest[1]}, Highest Separability",
        output_file=prefix + "embedding_visualization_pca_high.pdf",
    )

    # Visualize the embeddings using PCA for the lowest separability patch
    visualize_embeddings_pca(
        one_activations,
        other_activations,
        patch_lowest,
        title=f"Layer {patch_lowest[0]} - Patch {patch_lowest[1]}, Lowest Separability",
        output_file=prefix + "embedding_visualization_pca_low.pdf",
    )
    
    visualize_embeddings_pca(
        one_activations,
        other_activations,
        layer_mean_higest,
        title=f"Layer {layer_mean_higest} - Mean, Highest Separability",
        output_file=prefix + "embedding_visualization_pca_high_mean.pdf",
    )
    
    visualize_embeddings_pca(
        one_activations,
        other_activations,
        layer_mean_lowest,
        title=f"Layer {layer_mean_lowest} - Mean, Lowest Separability",
        output_file=prefix + "embedding_visualization_pca_low_mean.pdf",
    )
    
    visualize_embeddings_lda(
        one_activations,
        other_activations,
        patch_highest,
        title=f"Layer {patch_highest[0]} - Patch {patch_highest[1]}, Highest Separability",
        output_file=prefix + "embedding_visualization_lda_high.pdf",
    )
    
    visualize_embeddings_lda(   
        one_activations,
        other_activations,
        patch_lowest,
        title=f"Layer {patch_lowest[0]} - Patch {patch_lowest[1]}, Lowest Separability",
        output_file=prefix + "embedding_visualization_lda_low.pdf",
    )
    
    visualize_embeddings_lda(
        one_activations,
        other_activations,
        layer_mean_higest,
        title=f"Layer {layer_mean_higest} - Mean, Highest Separability",
        output_file=prefix + "embedding_visualization_lda_high_mean.pdf",
    )
    
    visualize_embeddings_lda(
        one_activations,
        other_activations,
        layer_mean_lowest,
        title=f"Layer {layer_mean_lowest} - Mean, Lowest Separability",
        output_file=prefix + "embedding_visualization_lda_low_mean.pdf",
    )


if __name__ == "__main__":

    sine_constant_activations = np.load("activations/sine_constant_activations.npy")[
        :, :256, :, :
    ]
    none_constant_activations = np.load("activations/none_constant_activations.npy")[
        :, :256, :, :
    ]

    compute_and_plot_separability(
        sine_constant_activations,
        none_constant_activations,
        prefix="default_model/",
    )
