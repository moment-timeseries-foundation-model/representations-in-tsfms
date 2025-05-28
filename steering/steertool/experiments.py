import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from pathlib import Path

from .moment import perturb_activations_MOMENT, get_activations_MOMENT
from .chronos import perturb_activations_Chronos, get_activations_Chronos, predict_Chronos
from .perturb import add
from .steering import get_steering_matrix
from .utils import load_dataset, get_sample_from_dataset
from .separability import compute_and_plot_separability, visualize_embeddings_pca, visualize_embeddings_lda


def extract_activations(dataset_path, model_type="moment", num_samples=20, device="cpu"):
    """
    Extract activations from a dataset for the specified model
    
    Parameters:
    -----------
    dataset_path : str
        Path to the parquet dataset
    model_type : str
        Model type ('moment' or 'chronos')
    num_samples : int
        Number of samples to use from the dataset
    device : str
        Device to run the model on ('cpu' or 'cuda')
        
    Returns:
    --------
    activations : numpy.ndarray
        The activations extracted from the model
    """
    logging.info(f"Extracting activations from {dataset_path} using {model_type} model")
    
    dataset = load_dataset(dataset_path, type="torch", device=device)
    
    if dataset.shape[0] > num_samples:
        logging.info(f"Limiting dataset from {dataset.shape[0]} to {num_samples} samples")
        dataset = dataset[:num_samples]
    
    if model_type.lower() == "moment":
        activations = get_activations_MOMENT(dataset, device=device)
        activations = activations.cpu().numpy() if device != "cpu" else activations.numpy()
        return activations
    
    elif model_type.lower() == "chronos":
        activations_encoder, activations_decoder = get_activations_Chronos(
            dataset.squeeze(1).cpu().numpy(), device=device
        )
        activations_encoder = activations_encoder.cpu().numpy() if device != "cpu" else activations_encoder.numpy()
        activations_decoder = activations_decoder.cpu().numpy() if device != "cpu" else activations_decoder.numpy()
        return activations_encoder
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_steering_experiment(
    source_dataset_path, 
    target_dataset_path,
    input_sample_path,
    input_sample_index=0,
    model_type="moment",
    method="mean",
    num_samples=20,
    alpha=1.0,
    beta=None,
    second_target_dataset_path=None,
    output_dir="results",
    device="cpu"
):
    """
    Run a steering experiment
    
    Parameters:
    -----------
    source_dataset_path : str
        Path to the source dataset (parquet)
    target_dataset_path : str
        Path to the target dataset (parquet)
    input_sample_path : str
        Path to the dataset containing the sample to steer
    input_sample_index : int
        Index of the sample to steer
    model_type : str
        Model type ('moment' or 'chronos')
    method : str
        Method to use for steering vector computation ('mean', 'median', 'lda')
    num_samples : int
        Number of samples to use from each dataset for steering vector computation
    alpha : float
        Steering strength
    beta : float, optional
        Second steering strength (for compositional steering)
    second_target_dataset_path : str, optional
        Path to the second target dataset (for compositional steering)
    output_dir : str
        Directory to save results
    device : str
        Device to run the model on ('cpu' or 'cuda')
        
    Returns:
    --------
    dict
        Dictionary containing the results of the experiment
    """
    logging.info(f"Running steering experiment: {source_dataset_path} -> {target_dataset_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    source_name = Path(source_dataset_path).stem
    target_name = Path(target_dataset_path).stem
    
    source_activations = extract_activations(source_dataset_path, model_type, num_samples, device)
    target_activations = extract_activations(target_dataset_path, model_type, num_samples, device)
    
    steering_vector = get_steering_matrix(source_activations, target_activations, method=method)
    
    if second_target_dataset_path and beta is not None:
        second_target_name = Path(second_target_dataset_path).stem
        second_target_activations = extract_activations(second_target_dataset_path, model_type, num_samples, device)
        second_steering_vector = get_steering_matrix(source_activations, second_target_activations, method=method)
        
        compositional_vector = (steering_vector * alpha + second_steering_vector * beta)
        steering_vector = compositional_vector
        output_prefix = f"{source_name}_to_{target_name}_{second_target_name}_{method}_alpha_{alpha}_beta_{beta}"
    else:
        output_prefix = f"{source_name}_to_{target_name}_{method}_alpha_{alpha}"
    
    input_dataset = pd.read_parquet(input_sample_path)
    input_sample = get_sample_from_dataset(input_dataset, input_sample_index)
    
    if model_type.lower() == "moment":
        non_perturbed_output = perturb_activations_MOMENT(input_sample, device=device)
        perturbed_output = perturb_activations_MOMENT(
            input_sample, 
            perturbation_fn=add, 
            perturbation_payload=alpha * steering_vector,
            device=device
        )
        non_perturbed_output = non_perturbed_output.flatten()
        perturbed_output = perturbed_output.flatten()
    
    elif model_type.lower() == "chronos":
        input_sample_np = input_sample.squeeze(1).cpu().numpy()
        non_perturbed_output = predict_Chronos(
            input_sample_np, 
            prediction_length=64,
            device=device
        )
        
        perturbed_output = perturb_activations_Chronos(
            input_sample_np[:,np.newaxis,:],
            prediction_length=64,
            device=device,
            perturbation_fn=add,
            perturbation_payload=alpha * steering_vector
        )
        
        non_perturbed_output = non_perturbed_output.cpu().numpy().flatten()
        perturbed_output = perturbed_output.cpu().numpy().flatten()
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    plot_path = os.path.join(output_dir, f"{output_prefix}.pdf")
    plot_imputed_signals_with_smoothing(non_perturbed_output, perturbed_output, save_path=plot_path)
    
    no_layers = source_activations.shape[0]
    layers_to_visualize = [0, no_layers//2, no_layers-1]
    
    for layer in layers_to_visualize:
        pca_path = os.path.join(output_dir, f"{output_prefix}_pca_layer_{layer}.pdf")
        visualize_embeddings_pca(
            source_activations,
            target_activations,
            layer,
            title=f"Layer {layer} - PCA Visualization",
            output_file=pca_path
        )
        
        lda_path = os.path.join(output_dir, f"{output_prefix}_lda_layer_{layer}.pdf")
        visualize_embeddings_lda(
            source_activations,
            target_activations,
            layer,
            title=f"Layer {layer} - LDA Visualization",
            output_file=lda_path
        )
    
    results = {
        "non_perturbed_output": non_perturbed_output,
        "perturbed_output": perturbed_output,
        "plot_path": plot_path
    }
    
    return results


def run_separability_analysis(
    dataset1_path,
    dataset2_path,
    analysis_type,
    model_type="moment",
    num_samples=20,
    output_dir="results",
    device="cpu"
):
    """
    Run a separability analysis experiment
    
    Parameters:
    -----------
    dataset1_path : str
        Path to the first dataset (parquet)
    dataset2_path : str
        Path to the second dataset (parquet)
    analysis_type : str
        Type of analysis ('constant-sine', 'trend', 'periodicity')
    model_type : str
        Model type ('moment' or 'chronos')
    num_samples : int
        Number of samples to use from each dataset
    output_dir : str
        Directory to save results
    device : str
        Device to run the model on ('cpu' or 'cuda')
        
    Returns:
    --------
    dict
        Dictionary containing the results of the analysis
    """
    logging.info(f"Running separability analysis: {dataset1_path} vs {dataset2_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset1_name = Path(dataset1_path).stem
    dataset2_name = Path(dataset2_path).stem
    output_prefix = f"{dataset1_name}_vs_{dataset2_name}_{analysis_type}"
    
    activations1 = extract_activations(dataset1_path, model_type, num_samples, device)
    activations2 = extract_activations(dataset2_path, model_type, num_samples, device)
    
    analysis_results = compute_and_plot_separability(
        activations1, 
        activations2, 
        prefix=os.path.join(output_dir, output_prefix)
    )
    
    return analysis_results


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
    imputed_normal_smoothed = pd.Series(imputed_normal).rolling(window=window_size, center=True).mean()
    imputed_perturbed_smoothed = pd.Series(imputed_perturbed).rolling(window=window_size, center=True).mean()

    sns.set(font_scale=2.0, style="ticks")
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rc('font', family='serif')

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(imputed_perturbed, label='Perturbed', color=palette[0], alpha=0.6, linewidth=2)
    ax.plot(imputed_normal, label='Non-Perturbed', color=palette[1], alpha=0.6, linewidth=2)

    ax.plot(imputed_perturbed_smoothed, label='Perturbed (Smoothed)', color=palette[0], linewidth=3)
    ax.plot(imputed_normal_smoothed, label='Non-Perturbed (Smoothed)', color=palette[1], linewidth=3)

    ax.set_xlabel("Timestep", fontsize=20)

    ax.legend(loc='best', fontsize=20)

    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace(' (Smoothed)', '') for label in labels]
    handles = [handles[2], handles[3]]
    ax.legend(handles, labels, loc='best', fontsize=20, frameon=True)
    
    plt.tight_layout(pad=2)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    plt.show() 