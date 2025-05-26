import warnings
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import json
import os
import yaml
import argparse
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from ...hooked_models import HookedMOMENT, HookedChronos, HookedMoirai
from ...datautils.data_generator import DataGenerator
from sklearn.model_selection import train_test_split
from momentfm.models.statistical_classifiers import fit_svm
import logging
import gc

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore", message="torch.utils._pytree._register_pytree_node is deprecated"
)
logging.basicConfig(
    filename="svm_classification.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


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

    if data_type == "classification":
        x_train, y_train = data_gen.get_classification_dataset(data_split="train")
        x_test, y_test = data_gen.get_classification_dataset(data_split="test")
        logging.debug(f"Train and test data shapes: {x_train.shape}, {x_test.shape}")

        return x_train, y_train, x_test, y_test
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def get_representations(config, x, model, model_name, is_train=True):
    random_seed = config["data_generation"]["random_seed"]
    cache_file = None
    if "none" not in config["cache"]:
        cache_file = os.path.join(
            config["cache"],
            f"{model_name.replace('/', '_')}_rs{random_seed}_{'train' if is_train else 'test'}.npy",
        )

    if os.path.exists(cache_file) and not config["force_recompute"]:
        return np.load(cache_file, allow_pickle=True)
    if "model_layernorm" in config["normalization"]:
        representations = model.get_layer_norms_representations(x)
    elif "model_final_layernorm" in config["normalization"]:
        model.get_encoder_representations(x, normalized_by_final_ln=True)
    elif "handcrafted_layernorm" or "none" in config["normalization"]:
        representations = model.get_encoder_representations(
            x, normalized_by_final_ln=False
        )

    representations = [r.cpu().detach().numpy() for r in representations]
    representations = np.array(representations)

    np.save(cache_file, representations) if cache_file is not None else None
    return representations


def train_and_evaluate_svm(x_train, y_train, x_test, y_test, config):
    svm = fit_svm(x_train, y_train)
    y_pred = svm.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report, svm


def plot_classification_results(results, config):
    plt.figure(figsize=(12, 6))
    for model_name, model_results in results.items():
        layers = list(model_results.keys())
        accuracies = [model_results[layer]["accuracy"] for layer in layers]
        plt.plot(layers, accuracies, marker="o", label=model_name)

    plt.title("SVM Classification Accuracy by Model and Layer Depth")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    max_no_layers = max(len(model_results) for model_results in results.values())
    for layer in range(max_no_layers):
        plt.axvline(x=layer, linestyle="--", color="gray", alpha=0.5)
    plt.xticks(range(max_no_layers))
    filename = f"svm_classification_depth_accuracies_rs{config['data_generation']['random_seed']}.png"
    plt.savefig(os.path.join(config["cache"], filename))
    plt.close()


def main(config_path):
    config = load_config(config_path)
    np.random.seed(config["data_generation"]["random_seed"])
    torch.manual_seed(config["data_generation"]["random_seed"])

    x_train, y_train, x_test, y_test = generate_data(config)

    results = {}
    total_models = sum(len(model_list) for model_list in config["models"].values())

    with tqdm(total=total_models, desc="Processing models", position=0) as pbar_models:
        for model_type, model_list in config["models"].items():
            model_class = globals()[f"Hooked{model_type}"]
            for model_name in model_list:
                model_config = prepare_config(
                    config["model_configs"][model_type], model_name
                )
                model = model_class(model_config)
                train_representations = get_representations(
                    config, x_train, model, model_name, is_train=True
                )
                test_representations = get_representations(
                    config, x_test, model, model_name, is_train=False
                )

                logging.debug(
                    f"Train and test representations shapes: {train_representations.shape}, {test_representations.shape}"
                )

                model_results = {}
                num_layers = len(train_representations)

                with tqdm(
                    total=num_layers,
                    desc=f"Layers for {model_name}",
                    position=0,
                    leave=False,
                ) as pbar_layers:
                    for layer, (train_rep, test_rep) in enumerate(
                        zip(train_representations, test_representations)
                    ):
                        logging.info(
                            f"Training and evaluating SVM for {model_name}, layer {layer}"
                        )

                        train_rep = np.mean(train_rep, axis=1)
                        test_rep = np.mean(test_rep, axis=1)
                        train_rep_flat = train_rep.reshape(train_rep.shape[0], -1)
                        test_rep_flat = test_rep.reshape(test_rep.shape[0], -1)

                        normalization_scheme = config["normalization"]
                        if "handcrafted_layernorm" in normalization_scheme:
                            train_rep_flat = (
                                train_rep_flat
                                - np.mean(train_rep_flat, axis=1, keepdims=True)
                            ) / (np.std(train_rep_flat, axis=1, keepdims=True) + 1e-8)
                            test_rep_flat = (
                                test_rep_flat
                                - np.mean(test_rep_flat, axis=1, keepdims=True)
                            ) / (np.std(test_rep_flat, axis=1, keepdims=True) + 1e-8)
                        elif (
                            "none"
                            or "model_layernorm"
                            or "model_final_layernorm" in normalization_scheme
                        ):
                            pass
                        else:
                            raise ValueError(
                                f"Unsupported normalization scheme: {normalization_scheme}"
                            )
                        logging.info(
                            f"l2 norm of x_train: {np.linalg.norm(train_rep_flat)}"
                        )
                        logging.info(
                            f"l2 norm of x_test: {np.linalg.norm(test_rep_flat)}"
                        )

                        accuracy, report, model = train_and_evaluate_svm(
                            train_rep_flat, y_train, test_rep_flat, y_test, config
                        )
                        model_results[layer] = {"accuracy": accuracy, "report": report}

                        pbar_layers.update(1)

                results[model_name] = model_results
                pbar_models.update(1)

                # Clean up memory
                del train_representations, test_representations, model
                gc.collect()
                torch.cuda.empty_cache()

    # Save results
    os.makedirs(config["cache"], exist_ok=True)
    results_filename = f"svm_classification_depth_results_rs{config['data_generation']['random_seed']}.json"
    with open(os.path.join(config["cache"], results_filename), "w") as f:
        json.dump(results, f, indent=2)

    # Plot results
    plot_classification_results(results, config)


def plot_cached_results(config_path):
    config = load_config(config_path)
    results_filename = f"svm_classification_depth_results_rs{config['data_generation']['random_seed']}.json"
    results_path = os.path.join(config["cache"], results_filename)

    if not os.path.exists(results_path):
        logging.error(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    plot_classification_results(results, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SVM classification experiment with configuration file."
    )
    logging.basicConfig(level=logging.INFO)
    parser.add_argument(
        "--config",
        type=str,
        default="tsfm_similarity/experiments/classification/config/default.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only plot the cached results without running the experiment",
    )
    args = parser.parse_args()

    if args.plot_only:
        plot_cached_results(args.config)
    else:
        main(args.config)
