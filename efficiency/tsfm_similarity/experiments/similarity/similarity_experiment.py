import warnings
import os

# Set environment variables to suppress warnings before any other imports
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::UserWarning'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Comprehensive warning suppression
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore", message="torch.utils._pytree._register_pytree_node is deprecated"
)
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*torch.utils._pytree._register_pytree_node.*"
)
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*torch.utils._pytree.register_pytree_node.*"
)

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import json
import yaml
import argparse
import matplotlib.pyplot as plt
from ...hooked_models import HookedMOMENT, HookedChronos, HookedMoirai
from ...datautils.data_generator import DataGenerator
from ...tools import fast_cka, robust_cca_similarity, activations_cosine_similarity
import logging


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def prepare_config(config, model_name):
    config_copy = config.copy()
    config_copy["model_version"] = model_name
    return config_copy


def generate_data(config):
    data_gen = DataGenerator(random_seed=config["data_generation"]["random_seed"])
    data_type = config["data_generation"]["type"]

    if data_type == "synthetic":
        params = config["data_generation"]["synthetic"]
        x, _ = data_gen.generate_synthetic_data(**params)
    elif data_type == "random":
        params = config["data_generation"]["random"]
        x = data_gen.generate_random_data(**params)
    elif data_type == "anomaly_detection":
        params = config["data_generation"]["anomaly_detection"]
        x, _ = data_gen.get_anomaly_detection_dataset(**params)
    elif data_type == "classification":
        params = config["data_generation"]["classification"]
        x, _ = data_gen.get_classification_dataset(**params)
    elif data_type == "forecasting":
        params = config["data_generation"]["forecasting"]
        x, _ = data_gen.get_forecasting_dataset(**params)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    return x


def get_representations_from_data(config):
    torch.manual_seed(config["data_generation"]["random_seed"])
    np.random.seed(config["data_generation"]["random_seed"])
    print(f"Using random seed: {config['data_generation']['random_seed']}")
    x = generate_data(config)
    logging.info(f"Generated data shape: {x.shape}")

    model_configs = []
    for model_type, model_list in config["models"].items():
        model_class = globals()[f"Hooked{model_type}"]
        for model_name in model_list:
            model_config = prepare_config(
                config["model_configs"][model_type], model_name
            )
            model_configs.append((model_class, model_config))

    representation_types = set()
    for model_class, model_config in tqdm(model_configs):
        print(f"Model: {model_config['model_version']}")
        model_name = model_config["model_version"].replace("/", "_")
        full_path = os.path.join(config["cache"], f"{model_name}.npy")

        if os.path.exists(full_path):
            logging.info(f"Loading representations for {model_name}")
            representations = np.load(full_path)
            representation_types.add((model_name, representations.shape))
            continue

        logging.info(f"Getting representations for {model_name}")
        model = model_class(model_config)
        try:
            representations = model.get_encoder_representations(x)
        except ValueError as e:
            logging.error(f"Error getting representations for {model_name}: {e}")
            continue

        representations = np.array([r.cpu().detach().numpy() for r in representations])
        representation_types.add((model_name, representations.shape))

        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        np.save(full_path, representations)

        del model
        del representations
        torch.cuda.empty_cache()

    json.dump(
        list(representation_types),
        open(os.path.join(config["cache"], "representation_types.json"), "w"),
    )


def get_similarities(representations_x, representations_y, metric):
    similarity_matrix = np.zeros((len(representations_x), len(representations_y)))
    for i, x in enumerate(representations_x):
        for j, y in enumerate(representations_y):
            similarity_matrix[i, j] = metric(x, y)
    similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (
        np.max(similarity_matrix) - np.min(similarity_matrix)
    )  # Normalization for better visualization
    return similarity_matrix


def plot_similarity(
    similarity_matrix,
    vmin=0,
    vmax=1,
    model1_name="model1",
    model2_name="model2",
    save=False,
    save_path=None,
    title="Similarity Matrix",
):

    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(12, 15))
    cax = ax.imshow(similarity_matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', pad=0.125)
    cbar.set_label("CKA Similarity", labelpad=10, fontsize=36)  # Adjust label fontsize
    cbar.ax.tick_params(labelsize=36)  # Adjust colorbar tick fontsize
    ticks = range(0, max(similarity_matrix.shape))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ticks_x = np.arange(1, similarity_matrix.shape[1])
    ticks_y = np.arange(1, similarity_matrix.shape[0])
    ax.set_xticks(ticks_x[::2])  # Show every other x tick
    ax.set_yticks(ticks_y[::2])  # Show every other y tick
    ax.invert_yaxis()

    def remap_name(name):
        if name == "AutonLab_MOMENT-1-large":
            return "MOMENT-Large"
        if name == "amazon_chronos-t5-tiny":
            return "Chronos-T5-Tiny"
        if name == "amazon_chronos-t5-mini":
            return "Chronos-T5-Mini"
        if name == "amazon_chronos-t5-small":
            return "Chronos-T5-Small"
        if name == "amazon_chronos-t5-base":
            return "Chronos-T5-Base"
        if name == "amazon_chronos-t5-large":
            return "Chronos-T5-Large"
        if name == "Salesforce_moirai-1.1-R-small":
            return "Moirai-1.1-R-Small"
        if name == "Salesforce_moirai-1.1-R-base":
            return "Moirai-1.1-R-Base"
        if name == "Salesforce_moirai-1.1-R-large":
            return "Moirai-1.1-R-Large"
        if name == "Salesforce_moirai-1.0-R-small":
            return "Moirai-1.0-R-Small"
        if name == "Salesforce_moirai-1.0-R-base":
            return "Moirai-1.0-R-Base"
        if name == "Salesforce_moirai-1.0-R-large":
            return "Moirai-1.0-R-Large"
        return name
    
    model1_name = remap_name(model1_name)
    model2_name = remap_name(model2_name)
    ax.set_xlabel(f"{model2_name} layer", fontsize=36)
    ax.set_ylabel(f"{model1_name} layer", fontsize=36)
    ax.set_title(title, fontsize=36)
    ax.tick_params(axis="both", which="major", labelsize=36)
    ax.grid(False)
    if save and save_path:
        plt.savefig(save_path)


def perform_similarity_analysis(config):
    supported_models = [
        model
        for model_type in config["models"]
        for model in config["models"][model_type]
    ]
    supported_models = [s.replace("/", "_") for s in supported_models]
    similarity_name = config["similarity"]
    if similarity_name == "cka":
        metric = fast_cka
    elif similarity_name == "cca":
        metric = robust_cca_similarity
    elif similarity_name == "cosine":
        metric = activations_cosine_similarity
    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_name}")
    model_representations = {
        model: np.load(os.path.join(config["cache"], f"{model}.npy"))
        for model in supported_models
        if os.path.exists(os.path.join(config["cache"], f"{model}.npy"))
    }

    model_representations_avg = {
        model: np.mean(model_representations[model], axis=2)
        for model in supported_models
        if os.path.exists(os.path.join(config["cache"], f"{model}.npy"))
    }

    for i, model1 in tqdm(
        enumerate(supported_models), total=len(supported_models), desc="Outer Loop"
    ):
        for j, model2 in enumerate(supported_models):
            if i <= j:
                if (
                    model_representations_avg.get(model1) is None
                    or model_representations_avg.get(model2) is None
                ):
                    continue
                similarity_matrix = get_similarities(
                    model_representations_avg[model1],
                    model_representations_avg[model2],
                    metric,
                )
                os.makedirs(config["results"], exist_ok=True)
                save_path = os.path.join(
                    config["results"],
                    f"tokens_similarity_{similarity_name}_{model1}_{model2}.pdf",
                )
                plot_similarity(
                    similarity_matrix,
                    vmin=0,
                    vmax=1,
                    model1_name=model1,
                    model2_name=model2,
                    save=True,
                    save_path=save_path,
                )


def main(config_path):
    config = load_config(config_path)
    get_representations_from_data(config)
    perform_similarity_analysis(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run similarity analysis with configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tsfm_similarity/experiments/similarity/config/default.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    main(args.config)
