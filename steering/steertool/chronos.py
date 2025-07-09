
from nnsight import NNsight
import torch
import numpy as np
from .utils import load_dataset
from .perturb import identity
import logging
import argparse
from chronos import ChronosPipeline


def get_Chronos(device="cpu"):
    torch.set_default_device(device)
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map=device,
    )
    pipeline.model.to(device)
    nnsight_model = NNsight(pipeline.model, device)
    logging.debug("Chronos model loaded")
    return nnsight_model, pipeline.tokenizer

def predict_Chronos(
    context, # (batch, sequence_length)
    prediction_length=64, # Time steps to predict, max 64
    num_samples=1, # Number of sample paths to predict
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    device="cpu",
) -> torch.Tensor:
    """
    Get forecasts for the given time series.

    Parameters
    ----------
    context
        Input series. This is either a 1D tensor, or a list
        of 1D tensors, or a 2D tensor whose first dimension
        is batch. In the latter case, use left-padding with
        ``torch.nan`` to align series of different lengths.
    prediction_length
        Time steps to predict. Defaults to what specified
        in ``self.model.config``.
    num_samples
        Number of sample paths to predict. Defaults to what
        specified in ``self.model.config``.
    temperature
        Temperature to use for generating sample tokens.
        Defaults to what specified in ``self.model.config``.
    top_k
        Top-k parameter to use for generating sample tokens.
        Defaults to what specified in ``self.model.config``.
    top_p
        Top-p parameter to use for generating sample tokens.
        Defaults to what specified in ``self.model.config``.
    limit_prediction_length
        Force prediction length smaller or equal than the
        built-in prediction length from the model. True by
        default. When true, fail loudly if longer predictions
        are requested, otherwise longer predictions are allowed.

    Returns
    -------
    samples
        Tensor of sample forecasts, of shape
        (batch_size, num_samples, prediction_length).
    """
    context_tensor = torch.tensor(context, dtype=torch.float32, device=device)
    
    chronos_model, chronos_tokenizer = get_Chronos(device=device)
    chronos_model = chronos_model._model

    predictions = []
    remaining = prediction_length

    while remaining > 0:
        token_ids, attention_mask, scale = chronos_tokenizer.context_input_transform(
            context_tensor
        )
        samples = chronos_model(
            token_ids.to(device),
            attention_mask.to(device),
            min(remaining, prediction_length),
            num_samples,
            temperature,
            top_k,
            top_p,
        )
        prediction = chronos_tokenizer.output_transform(
            samples.to(scale.device), scale
        )

        predictions.append(prediction)
        remaining -= prediction.shape[-1]

        if remaining <= 0:
            break

        context_tensor = torch.cat(
            [context_tensor, prediction.median(dim=1).values], dim=-1
        )

    return torch.cat(predictions, dim=-1)

def get_activations_Chronos(
    context, # (batch, sequence_length)
    prediction_length=1, # Time steps to predict, max 64
    num_samples=1, # Number of sample paths to predict
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    device="cpu",
) -> torch.Tensor:
    """
    Get forecasts for the given time series.

    Parameters
    ----------
    context
        Input series. This is either a 1D tensor, or a list
        of 1D tensors, or a 2D tensor whose first dimension
        is batch. In the latter case, use left-padding with
        ``torch.nan`` to align series of different lengths.
    prediction_length
        Time steps to predict. Defaults to what specified
        in ``self.model.config``.
    num_samples
        Number of sample paths to predict. Defaults to what
        specified in ``self.model.config``.
    temperature
        Temperature to use for generating sample tokens.
        Defaults to what specified in ``self.model.config``.
    top_k
        Top-k parameter to use for generating sample tokens.
        Defaults to what specified in ``self.model.config``.
    top_p
        Top-p parameter to use for generating sample tokens.
        Defaults to what specified in ``self.model.config``.
    limit_prediction_length
        Force prediction length smaller or equal than the
        built-in prediction length from the model. True by
        default. When true, fail loudly if longer predictions
        are requested, otherwise longer predictions are allowed.

    Returns
    -------
    samples
        Tensor of sample forecasts, of shape
        (batch_size, num_samples, prediction_length).
    """
    chronos_model, chronos_tokenizer = get_Chronos(device=device)
    context_tensor = torch.tensor(context, dtype=torch.float32, device=device)

    token_ids, attention_mask, scale = chronos_tokenizer.context_input_transform(
        context_tensor
    )
    
    activations_encoder = []
    activations_decoder = []
    logging.debug("Extracting activations for Chronos model")
    with chronos_model.trace(token_ids.to(device),
        attention_mask.to(device),
        prediction_length,
        num_samples,
        temperature,
        top_k,
        top_p,) as trace:
        for i, layer in enumerate(chronos_model.model.encoder.block):
            activations_encoder.append(layer.output[0].clone().save())
        for i, layer in enumerate(chronos_model.model.decoder.block):
            activations_decoder.append(layer.output[0].clone().save())
    logging.debug("Activations extracted")

    activations_encoder = torch.stack(activations_encoder, dim=0)
    activations_decoder = torch.stack(activations_decoder, dim=0)
    return activations_encoder, activations_decoder

def perturb_activations_Chronos(
    dataset, # (batch, channel, sequence_length)
    prediction_length=64, # Time steps to predict, max 64
    num_samples=1, # Number of sample paths to predict
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    device="cpu",
    perturbation_fn=identity,
    perturbation_payload=torch.ones(24, 513, 1024),
    layer_indices=list(range(24)),
    token_indices=list(range(513)),
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
    if not isinstance(perturbation_payload, torch.Tensor):
        perturbation_payload = torch.tensor(perturbation_payload, device=device)
    batch, channels, seq_len = dataset.shape
    dataset = dataset.reshape(batch, seq_len)
    context_tensor = torch.tensor(dataset, dtype=torch.float32, device=device)
    model_outputs = None
    
    chronos_model, chronos_tokenizer = get_Chronos(device=device)
    
    predictions = []
    remaining = prediction_length
    
    while remaining > 0:
        token_ids, attention_mask, scale = chronos_tokenizer.context_input_transform(
            context_tensor
        )

        with chronos_model.trace(token_ids,
            attention_mask,
            min(remaining, prediction_length),
            num_samples,
            temperature,
            top_k,
            top_p,) as trace:
            for block_idx, transformer_block in enumerate(chronos_model.model.encoder.block):
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
            model_outputs = chronos_model.output.save()
        prediction = chronos_tokenizer.output_transform(
                    model_outputs.to(scale.device), scale
        )
        
        predictions.append(prediction)
        remaining -= prediction.shape[-1]
        
        if remaining <= 0:
            break
        
        context_tensor = torch.cat(
            [context_tensor, prediction.median(dim=1).values], dim=-1
        )
        
    prediction_final = torch.cat(predictions, dim=-1)
    
    logging.debug(f"Activations perturbed, output shape:{prediction_final.shape}")
    return prediction_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract activations from Chronos model"
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
    dataset = load_dataset(dataset_path=args.dataset, type="torch", device=device).squeeze(1)

    activations, _ = get_activations_Chronos(dataset, device=device)
    logging.debug(f"Activations shape: {activations.shape}")

    output_file = args.dataset.split("/")[-1].split(".")[0] + "_activations.npy"
    output_file = f"activations_chronos/{output_file}"

    logging.debug(f"Saving activations to {output_file}")
    activations = (
        activations.cpu().numpy() if device is not "cpu" else activations.numpy()
    )
    np.save(output_file, activations)
    logging.info(f"Activations saved to {output_file}")
