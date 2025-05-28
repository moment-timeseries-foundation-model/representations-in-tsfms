import numpy as np
import torch
import pandas as pd
import logging


def load_dataset(dataset_path, type="pandas", device="cpu"):
    """
    Load a dataset to specific type (pandas, numpy, torch) and device
    """
    dataset = pd.read_parquet(dataset_path)
    logging.debug(f"Dataset loaded: {dataset.shape}")

    if type == "pandas":
        return dataset
    elif type in ["numpy", "torch"]:
        dataset_len = len(dataset)
        X_np = np.array(
            [dataset["series"].values[i] for i in range(dataset_len)]
        ).reshape(dataset_len, 1, -1)
        logging.debug(f"Dataset shape: {X_np.shape}")
        return X_np if type == "numpy" else torch.tensor(X_np, device=device).float()
    else:
        raise ValueError("Invalid type")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.debug(f"Seed set to {seed}")
    
def get_sample_from_dataset(dataset, sample_idx):
    return torch.tensor(dataset['series'].values[sample_idx].reshape(-1, 1, 512), dtype=torch.float32)
