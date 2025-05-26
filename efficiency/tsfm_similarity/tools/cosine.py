import numpy as np


def activations_cosine_similarity(acts1: np.ndarray, acts2: np.ndarray) -> float:
    """
    Calculate the average cosine similarity between two sets of activations.

    Args:
    acts1 (np.ndarray): First set of activations
    acts2 (np.ndarray): Second set of activations

    Returns:
    float: Average cosine similarity between the two sets of activations
    """
    if acts1.shape[1] != acts2.shape[1]:
        raise ValueError("Model dimensions do not match")
    if acts1.shape[0] != acts2.shape[0]:
        raise ValueError("Batch sizes do not match")
    dot_products = np.sum(acts1 * acts2, axis=1)
    norms1 = np.linalg.norm(acts1, axis=1)
    norms2 = np.linalg.norm(acts2, axis=1)
    similarities = np.mean(dot_products) / (norms1 * norms2 + 1e-8)
    return similarities
