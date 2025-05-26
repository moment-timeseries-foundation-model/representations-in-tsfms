from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from tqdm import tqdm  # Import tqdm for progress tracking

def compute_lda_steering_vector(one_data, other_data):
    """
    Compute the vector pointing from one class to another along the LDA discriminant direction.
    """
    # Combine the two sets of activations into one dataset
    X = np.concatenate((one_data, other_data), axis=0)

    # Create labels for LDA (0 for 'one', 1 for 'other')
    y = np.concatenate((np.zeros(one_data.shape[0]), np.ones(other_data.shape[0])))

    # Fit the LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # Get the LDA discriminant vector (direction in feature space)
    lda_direction = lda.coef_.flatten()

    return lda_direction


def get_steering_matrix(one_activations, other_activations, method="median", n_jobs=-1):
    """
    Get the steering matrix from one_activations to other_activations.
    Parallelized for 'lda' method, with progress tracking.
    """
    layer, batch, patch, features = one_activations.shape

    if method == "median":
        one_median = np.median(one_activations, axis=1)
        other_median = np.median(other_activations, axis=1)
        steering_matrix = other_median - one_median

    elif method == "mean":
        one_mean = np.mean(one_activations, axis=1)
        other_mean = np.mean(other_activations, axis=1)
        steering_matrix = other_mean - one_mean

    elif method == "lda":
        # Initialize the steering matrix
        steering_matrix = np.zeros((layer, patch, features))

        # Function to compute LDA steering vector for each (layer, patch)
        def compute_for_patch(l, p):
            one_data = one_activations[l, :, p, :]
            other_data = other_activations[l, :, p, :]
            return l, p, compute_lda_steering_vector(one_data, other_data)

        # Create a list of tasks for parallel processing
        tasks = [(l, p) for l in range(layer) for p in range(patch)]

        # Use tqdm to track progress
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_for_patch)(l, p) for l, p in tqdm(tasks, desc="Computing LDA Steering Vectors")
        )

        # Fill in the steering matrix with the LDA steering vectors from the results
        for l, p, lda_vector in results:
            steering_matrix[l, p, :] = lda_vector

    # Save the steering matrix
    np.save(f"steering_matrix_{method}.npy", steering_matrix)
    return steering_matrix