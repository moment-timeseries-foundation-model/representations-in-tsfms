from momentfm import MOMENTPipeline
from nnsight import NNsight
import torch
import numpy as np
from src.utils import load_dataset
from src.perturb import identity
import logging
import argparse


def get_MOMENT(device="cpu"):
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction", "device": device},
    )
    model.init()
    model.to(device)
    nnsight_model = NNsight(model, device)
    logging.debug("MOMENT model loaded")
    return nnsight_model


def perturb_activations_MOMENT(
    dataset,
    perturbation_fn=identity,
    perturbation_payload=torch.ones(24, 64, 1024),
    save_activations=False,
    device="cpu",
    layer_indices=list(range(24)),
    token_indices=list(range(64)),
):
    """
    Perturb activations in specified layers and tokens using NNsight and a custom perturbation function.

    Args:
        dataset (torch.Tensor): Input data tensor of shape (batch_size, channels, sequence_length).
        device (torch.device): Device to run the model on ('cpu' or 'cuda:x').
        layer_indices (int or list of int): Indices of layers to perturb.
        token_indices (int or list of int): Indices of tokens to perturb.
        perturbation_fn (callable): Function to apply to the activations.

    Returns:
        torch.Tensor: The output of the model after perturbation.
    """
    # Ensure layer_indices and token_indices are lists
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    if isinstance(token_indices, int):
        token_indices = [token_indices]

    batch, channels, seq_len = dataset.shape
    input_mask = create_prediction_mask_MOMENT(batch, seq_len, seq_len, device)
    moment_model = get_MOMENT(device=device)
    model_outputs = None

    with moment_model.trace(dataset, input_mask=input_mask) as trace:
        for block_idx, transformer_block in enumerate(moment_model.encoder.block):
            if block_idx in layer_indices:
                current_layer_activations = transformer_block.layer[
                    -1
                ].output.clone()  # (1, 64, 1024) (batch, tokens, features)
                for token_idx in token_indices:
                    current_layer_activations[:, token_idx, :] = perturbation_fn(
                        current_layer_activations[:, token_idx, :], perturbation_payload[block_idx, token_idx, :]
                    ) # will get broadcasted along the batch dimension, if necessary, automatically
                transformer_block.layer[-1].output = current_layer_activations
                logging.debug(f"Perturbed layer {block_idx}, tokens {token_indices}")
        model_outputs = moment_model.head.output.save()
    model_outputs_numpy = model_outputs.value.detach().numpy()
    logging.debug(f"Activations perturbed, output shape:{model_outputs_numpy.shape}")
    return model_outputs_numpy


def create_prediction_mask_MOMENT(
    batch_size, sequence_length, mask_final_n_samples, device="cpu"
):
    if mask_final_n_samples <= 1:
        mask_final_n_samples = int(sequence_length * mask_final_n_samples)
    mask = torch.ones(1, sequence_length, dtype=torch.float32, device=device)
    mask[:, :-mask_final_n_samples] = 0
    mask = mask.repeat(batch_size, 1)
    logging.debug(f"Prediction mask created: {mask.shape}")
    return mask


def get_activations_MOMENT(dataset, device="cpu"):
    batch, channels, seq_len = dataset.shape
    input_mask = create_prediction_mask_MOMENT(batch, seq_len, seq_len, device)
    moment_model = get_MOMENT(device=device)
    activations = []
    logging.debug("Extracting activations for MOMENT model")
    with moment_model.trace(dataset, input_mask=input_mask) as trace:
        for i, layer in enumerate(moment_model.encoder.block):
            activations.append(layer.output[0].clone().save())
    logging.debug("Activations extracted")
    return torch.stack(activations, dim=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract activations from MOMENT model"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (e.g., 'cpu', 'cuda')",
    )
    parser.add_argument("--log", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(level=args.log)

    device = torch.device(args.device)
    dataset = load_dataset(dataset_path=args.dataset, type="torch", device=device)

    activations = get_activations_MOMENT(dataset, device)
    logging.debug(f"Activations shape: {activations.shape}")

    output_file = args.dataset.split("/")[-1].split(".")[0] + "_activations.npy"
    output_file = f"activations_moment/{output_file}"

    logging.debug(f"Saving activations to {output_file}")
    activations = (
        activations.cpu().numpy() if device is not "cpu" else activations.numpy()
    )
    np.save(output_file, activations)
    logging.info(f"Activations saved to {output_file}")
