from momentfm import MOMENTPipeline
import torch
from .utils import load_dataset
from .perturb import identity
import logging
import argparse


def perturb_activations_MOMENT(
    dataset,
    perturbation_fn=identity,
    perturbation_payload=torch.ones(24, 64, 1024),
    device="cpu",
    layer_indices=list(range(24)),
    token_indices=list(range(64)),
):
    """
    Perturb activations in specified layers and tokens using forward and backward hooks.
    
    Args:
        dataset (torch.Tensor): Input data tensor of shape (batch_size, channels, sequence_length).
        perturbation_fn (callable): Function to apply to the activations.
        perturbation_payload (torch.Tensor): Tensor of perturbation values.
        save_activations (bool): Whether to save activations.
        device (str): Device to run the model on ('cpu' or 'cuda').
        layer_indices (list): Indices of layers to perturb.
        token_indices (list): Indices of tokens to perturb.
        
    Returns:
        numpy.ndarray: The output of the model after perturbation.
    """
    # Ensure layer_indices and token_indices are lists
    if isinstance(layer_indices, int):
        layer_indices = [layer_indices]
    if isinstance(token_indices, int):
        token_indices = [token_indices]
    
    # Ensure device is a torch device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    
    # Move dataset to the correct device
    dataset = dataset.to(device)
    
    # Move perturbation payload to the correct device
    perturbation_payload = torch.tensor(perturbation_payload, device=device)
    
    batch, channels, seq_len = dataset.shape
    input_mask = create_prediction_mask_MOMENT(batch, seq_len, seq_len, device)
    
    # Create the model
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction", "device": device},
    )
    model.init()
    model.to(device)
    
    # Dictionary to store perturbation hooks
    perturbation_hooks = []
    
    # Define a perturbation hook function
    def perturbation_hook(block_idx):
        def hook(module, input, output):
            # Only perturb if this layer is in the target layers
            if block_idx in layer_indices:
                perturbed_output = output.clone()
                for token_idx in token_indices:
                    if token_idx < perturbed_output.shape[1]:  # Check token index is valid
                        # Ensure perturbation is on the same device as output
                        current_perturbation = perturbation_payload[block_idx, token_idx, :].to(output.device)
                        perturbed_output[:, token_idx, :] = perturbation_fn(
                            perturbed_output[:, token_idx, :],
                            current_perturbation
                        )
                logging.debug(f"Perturbed layer {block_idx}, tokens {token_indices}")
                return perturbed_output
            return output
        return hook
    
    # Register hooks for each transformer block
    for i, block in enumerate(model.encoder.block):
        # Register a forward hook on the last layer of each block
        hook = block.layer[-1].register_forward_hook(perturbation_hook(i))
        perturbation_hooks.append(hook)
    
    # Forward pass with perturbations
    with torch.no_grad():
        outputs = model(x_enc=dataset, input_mask=input_mask)
    
    # Remove all hooks
    for hook in perturbation_hooks:
        hook.remove()
    
    # Convert outputs to numpy
    model_outputs_numpy = outputs.reconstruction.detach().cpu().numpy()
    logging.debug(f"Activations perturbed, output shape: {model_outputs_numpy.shape}")
    
    return model_outputs_numpy


def create_prediction_mask_MOMENT(
    batch_size, sequence_length, mask_final_n_samples, device="cpu"
):
    if not isinstance(device, torch.device):
        device = torch.device(device)
        
    if mask_final_n_samples <= 1:
        mask_final_n_samples = int(sequence_length * mask_final_n_samples)
    mask = torch.ones(1, sequence_length, dtype=torch.float32, device=device)
    mask[:, :-mask_final_n_samples] = 0
    mask = mask.repeat(batch_size, 1)
    logging.debug(f"Prediction mask created: {mask.shape}")
    return mask


def get_activations_MOMENT(dataset, device="cpu"):
    """
    Extract activations from all layers of the MOMENT model using a hook-based approach.
    
    Args:
        dataset (torch.Tensor): Input data tensor of shape (batch_size, channels, sequence_length).
        device (str): Device to run the model on ('cpu' or 'cuda').
        
    Returns:
        torch.Tensor: Stacked activations from all layers.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    
    dataset = dataset.to(device)
    
    batch, channels, seq_len = dataset.shape
    input_mask = create_prediction_mask_MOMENT(batch, seq_len, seq_len, device)
    logging.debug("Extracting activations for MOMENT model")
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction", "device": device},
    )
    model.init()
    model.to(device)
    
    all_activations = []
    
    def hook_fn(module, input, output):
        all_activations.append(output.detach().clone())
    
    hooks = []
    for i, block in enumerate(model.encoder.block):
        hooks.append(block.layer[-1].register_forward_hook(hook_fn))
    
    with torch.no_grad():
        _ = model(x_enc=dataset, input_mask=input_mask)
    
    for hook in hooks:
        hook.remove()
    
    stacked_activations = torch.stack(all_activations, dim=0)
    logging.debug(f"Total activations shape: {stacked_activations.shape}")
    
    return stacked_activations


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
    
    logging.info("Activations extracted successfully")
