#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import logging

def plot_random_samples(X, n_samples=3, save_path=None, title=None):
    """
    Plot random samples from time-series data X.
    
    Parameters:
    - X: numpy array or pandas DataFrame of time-series data
    - n_samples: number of random samples to plot
    - save_path: if provided, saves the plot to this path
    - title: optional title for the figure
    """
    # Randomly select n_samples from the dataset
    sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    
    # Create a figure and axes for subplots in a row
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))

    # Handle case where n_samples=1
    if n_samples == 1:
        axes = [axes]
        
    # Loop through the sample indices and plot each sample on its respective axis
    for i, idx in enumerate(sample_indices):
        y_vals = X[idx].squeeze()
        axes[i].plot(y_vals, linewidth=2)
        axes[i].tick_params(axis='both', which='major', labelsize=16)
        
        # Remove extra ticks from top and right sides for a clean look
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

        # Set labels and title with larger font sizes
        axes[i].set_xlabel("Time", fontsize=18)
        axes[i].set_ylabel("Value", fontsize=18) if i == 0 else axes[i].set_ylabel("")
        axes[i].set_title(f"Sample {i+1}", fontsize=20)

    # Add a main title if provided
    if title:
        fig.suptitle(title, fontsize=22, y=1.05)
        
    # Adjust layout for better spacing between plots
    plt.tight_layout(pad=2)

    # Save the figure if a path is provided
    if save_path:
        logging.info(f"Saving samples plot to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_attribute_distributions(dataset, save_path=None, title=None):
    """
    Plot distributions of various attributes from the dataset.
    
    Parameters:
    - dataset: pandas DataFrame containing the data
    - save_path: if provided, saves the plot to this path
    - title: optional title for the figure
    """
    # Set the overall aesthetic for the plots with larger font sizes for readability
    sns.set(font_scale=2.0, style="ticks")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    
    # Define a consistent color for all the plots
    color = sns.color_palette()[0]
    
    # Create a figure with 2 rows and 2 columns for the 4 attributes
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot attribute distributions
    try:
        # 1. Pattern type distribution
        sns.countplot(x='seasonality_type', data=dataset, ax=axes[0, 0], color=color, alpha=1.0)
        axes[0, 0].set_title("Pattern Type Distribution", fontsize=22)
        axes[0, 0].set_xlabel("Pattern Type", fontsize=20)
        axes[0, 0].set_ylabel("Count", fontsize=20)

        # 2. Y-intercept distribution
        sns.histplot(x='trend_intercept', data=dataset, ax=axes[0, 1], color=color, alpha=1.0)
        axes[0, 1].set_title("Y-Intercept Distribution", fontsize=22)
        axes[0, 1].set_xlabel("Y-Intercept", fontsize=20)
        axes[0, 1].set_ylabel("Frequency", fontsize=20)

        # 3. Pattern amplitude distribution
        sns.histplot(x='seasonality_amplitude', data=dataset, ax=axes[1, 0], color=color, alpha=1.0)
        axes[1, 0].set_title("Pattern Amplitude Distribution", fontsize=22)
        axes[1, 0].set_xlabel("Pattern Amplitude", fontsize=20)
        axes[1, 0].set_ylabel("Frequency", fontsize=20)

        # 4. Pattern period distribution
        sns.histplot(x='seasonality_period', data=dataset, ax=axes[1, 1], color=color, alpha=1.0)
        axes[1, 1].set_title("Pattern Period Distribution", fontsize=22)
        axes[1, 1].set_xlabel("Pattern Period", fontsize=20)
        axes[1, 1].set_ylabel("Frequency", fontsize=20)
    except Exception as e:
        logging.warning(f"Error plotting attribute distributions: {e}")
        plt.close(fig)
        return

    # Add a main title if provided
    if title:
        fig.suptitle(title, fontsize=24, y=1.05)

    # Adjust tick labels for better visibility
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Adjust layout for a clean look with enough space between plots
    plt.tight_layout(pad=3.0)

    # Save the figure if a path is provided
    if save_path:
        logging.info(f"Saving attribute distributions plot to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
        plt.close(fig)
    else:
        plt.show()

def visualize_dataset(dataset_path, output_dir="data_visualizations", n_samples=3, name=None):
    """
    Visualize a dataset by plotting random samples and attribute distributions.
    
    Parameters:
    - dataset_path: Path to the dataset file (.parquet)
    - output_dir: Directory to save the visualizations
    - n_samples: Number of random samples to plot
    - name: Optional name for the dataset (defaults to the filename)
    """
    # Load the dataset
    try:
        dataset = pd.read_parquet(dataset_path)
        logging.info(f"Loaded dataset from {dataset_path}")
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_path}: {e}")
        return
    
    # Use the filename as the name if not provided
    if name is None:
        name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the series data
    X = np.array([dataset['series'].values[i] for i in range(len(dataset))])
    
    # Plot random samples
    samples_path = os.path.join(output_dir, f"timeseries_samples_{name}.pdf")
    plot_random_samples(
        X, n_samples=n_samples, save_path=samples_path, title=f"Dataset: {name}"
    )
    
    # Plot attribute distributions
    stats_path = os.path.join(output_dir, f"appendix_stats_{name}.pdf")
    plot_attribute_distributions(
        dataset, save_path=stats_path, title=f"Dataset: {name}"
    )
    
    logging.info(f"Visualizations for {name} saved to {output_dir}")

def visualize_all_datasets(dataset_dir="datasets", output_dir="data_visualizations", n_samples=3, skip_ecg=True):
    """
    Visualize all datasets in a directory.
    
    Parameters:
    - dataset_dir: Directory containing dataset files (.parquet)
    - output_dir: Directory to save the visualizations
    - n_samples: Number of random samples to plot
    - skip_ecg: Whether to skip ECG dataset files
    """
    # List all parquet files in the directory
    try:
        dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith('.parquet')]
        
        # Skip ECG files if requested
        if skip_ecg:
            dataset_files = [f for f in dataset_files if 'ecg' not in f.lower()]
            
        logging.info(f"Found {len(dataset_files)} dataset files in {dataset_dir}")
    except Exception as e:
        logging.error(f"Error listing datasets in {dataset_dir}: {e}")
        return
    
    # Visualize each dataset
    for dataset_file in dataset_files:
        dataset_path = os.path.join(dataset_dir, dataset_file)
        name = os.path.splitext(dataset_file)[0]
        visualize_dataset(dataset_path, output_dir, n_samples, name)
    
    logging.info(f"All visualizations saved to {output_dir}") 