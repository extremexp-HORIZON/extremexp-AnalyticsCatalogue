import numpy as np
import pandas as pd
import math


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def apply_smote_class(samples: pd.DataFrame, n_samples: int, k: int = 5, random_state: int = None) -> pd.DataFrame:
    """
    :param samples: pandas DataFrame.
    :param n_samples: Number of synthetic samples to generate.
    :param k: Number of nearest neighbors.
    :param random_state: Seed for reproducibility.
    :return: DataFrame containing n_samples synthetic samples.
    """
    np.random.seed(random_state)

    num_existing_samples, _ = samples.shape
    N = math.ceil(n_samples/num_existing_samples)  # this will be our multiplier
    new_data = []

    # Extract feature columns for calculations
    feature_columns = samples.values
 
    # Find k nearest neighbors for each sample.
    for i in range(num_existing_samples):
        distances = np.apply_along_axis(euclidean_distance, 1, feature_columns, feature_columns[i])
        nn_indices = np.argsort(distances)[1:k+1]

        for n in range(N): 
            # Randomly choose one of the k nearest neighbors.
            nn = feature_columns[np.random.choice(nn_indices)]
            diff = nn - feature_columns[i]
            
            # Generate a random value between 0 and 1 and multiply with diff.
            gap = np.random.rand()
            new_point = feature_columns[i] + gap * diff
            new_data.append(new_point)

    # Create a new DataFrame from synthetic samples with the desired amount of samples
    new_samples_df = pd.DataFrame(new_data, columns=samples.columns).sample(n_samples)
 
    return new_samples_df


def smote_augmentation(df: pd.DataFrame, label: str, n_samples: int, random_state: int = None) -> pd.DataFrame:
    """
    Applies SMOTE to a given dataset.
    Parameters:
    - df (pd.DataFrame): The input dataset containing features and labels.
    Returns:
    - synthetic_df (pd.DataFrame): A dataframe containing synthetic samples.
    """
    
    #print(f"DEBUG: SMOTE {label}")
    majority_label = df[label].mode()[0]
    synthetic_dfs = [] 
    for target, group in df.groupby(label):
        # Calculate N for SMOTE
 
        if target == majority_label or n_samples == 0:
            continue

        # Apply SMOTE to the group
 
        x = group.drop(label, axis=1)
        synthetic_samples = apply_smote_class(samples=x, n_samples=n_samples, random_state=random_state)
        synthetic_samples[label] = target
        synthetic_dfs.append(synthetic_samples)

    synthetic_df = pd.concat(synthetic_dfs, ignore_index=True)

    return synthetic_df
