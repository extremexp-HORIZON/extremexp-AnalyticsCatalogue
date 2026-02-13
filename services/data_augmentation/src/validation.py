import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.spatial.distance import jensenshannon


def compute_cosine_similarity(dataset_A: np.ndarray, dataset_B: np.ndarray) -> dict:
    num_sequences = dataset_A.shape[0]
    feature_similarities = np.zeros(dataset_A.shape[2])
    
    for k in range(dataset_A.shape[2]):  # Iterate over features
        similarities = []
        for i in range(num_sequences):
            for j in range(num_sequences):
                sim = 1 - cosine(dataset_A[i, :, k], dataset_B[j, :, k])
                similarities.append(sim)
        feature_similarities[k] = np.mean(similarities)
    
    return {"feature_means": feature_similarities, "global_mean": np.mean(feature_similarities)}


def compute_jensen_shannon_divergence(dataset_A: np.ndarray, dataset_B: np.ndarray) -> dict:
    num_sequences = dataset_A.shape[0]
    feature_divergences = np.zeros(dataset_A.shape[2])
    
    for k in range(dataset_A.shape[2]):  # Iterate over features
        divergences = []
        for i in range(num_sequences):
            for j in range(num_sequences):
                p = dataset_A[i, :, k] + 1e-10  # Add small constant to avoid zero probabilities
                q = dataset_B[j, :, k] + 1e-10
                p /= np.sum(p)  # Normalize to probability distribution
                q /= np.sum(q)
                div = jensenshannon(p, q)
                divergences.append(div)
        feature_divergences[k] = np.mean(divergences)
    
    return {"feature_means": feature_divergences, "global_mean": np.mean(feature_divergences)}


def compute_time_series_similarity(dataset_A: np.ndarray, dataset_B: np.ndarray, distance: str) -> dict:
    if distance == "cosine_similarity":
        return compute_cosine_similarity(dataset_A, dataset_B)
    elif distance == "jensen_shannon_divergence":
        return compute_jensen_shannon_divergence(dataset_A, dataset_B)
    else:
        raise ValueError("Unsupported distance metric")


if __name__ == "__main__":
    dataset_A = np.random.normal(0, 1, (50, 300, 2))
    dataset_B = np.random.normal(0, 1, (50, 300, 2))
    distance = "jensen_shannon_divergence"  # or "cosine_similarity"
    
    result = compute_time_series_similarity(dataset_A, dataset_B, distance)
    print(result)
