# This script is used to generate the reedy approach conssting in splitijng the data in several buckets and iteritevaly apply
# DA on those buckets to create a better dataset

import re
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split

import src.data_augmentation.ctgan_algorithm as da_ctgan
import src.data_augmentation.smote_algorithm as da_smote
import src.data_augmentation.adasyn_algorithm as da_adasyn


SEED = 43


def bucketizer(dataset: pd.DataFrame) -> list:
    """
    Creates a list of subsets (buckets) from the given dataset for data augmentation purposes.

    Parameters:
    - dataset (pd.DataFrame): The input dataset to be divided into subsets.

    Returns:
    - subsets (list): A list of subsets (buckets), each containing a fixed number of samples for data augmentation.
    """

    N = len(dataset[dataset["label"] == 1])
    # Set the number of subsets (M) and size of each subset (L)
    M = int(0.10 * N)
    L = int(N / M)*10
    #M = 5
    #L = 100

    subsets = [dataset.sample(n=L, replace=False) for _ in range(M)]
    print(f"[DEBUG]: Bukets Made {M} of length {L}")
    return subsets


def dataset_similarity(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:
    """
    Calculates the Wasserstein distance-based similarity between two datasets.

    Parameters:
    - dataset_A (pd.DataFrame): The first dataset for comparison.
    - dataset_B (pd.DataFrame): The second dataset for comparison.

    Returns:
       - mean_w_distance (float): The mean Wasserstein distance between corresponding feature distributions of the two datasets.
    """

    # Calculate the distance for each feature
    w_distances = {col: wasserstein_distance(dataset_A[col], dataset_B[col]) for col in dataset_A.columns}
    # Calculate the mean Wasserstein distance
    mean_w_distance = sum(w_distances.values()) / len(w_distances)

    return mean_w_distance


def data_augmentation(dataset: pd.DataFrame) -> dict:
    """
    Applies data augmentation techniques to a given dataset.

    Parameters:
    - dataset (pd.DataFrame): The input dataset containing features and labels.

    Returns:
    - synth_data (dict): A dictionary containing keys as augmentation algorithms and values as dataframes with augmented samples.
    """
    #print("------------------------------------------------------------------------")
    #print(dataset["label"].value_counts())

    synth_data = {}
    most_common_label = dataset['label'].mode()[0]
    number_samples = len(dataset[dataset['label'] != most_common_label])

    synth_data["SMOTE"] = da_smote.smote_augmentation(dataset, "label", number_samples)
    #synth_data["ADASYN"] = da_adasyn.adasyn_augmentation(dataset, "label", number_samples)
    #synth_data["CTGAN"] = da_ctgan.ctgan_augmentation(dataset, "label", number_samples)

    for key, value in synth_data.items():
        #print(f"[DEBUG]: ALGORITHM: {key}, SHAPE: {value.shape}")
        #print(value["label"].value_counts())
        continue

    return synth_data


def filter_augmented_samples(augmented_bucket: dict, distances: list, threshold: float) -> dict:
    """
    Filters augmented samples based on the provided distances and a given threshold.

    Parameters:
    - augmented_bucket (dict): A dictionary where keys represent augmentation algorithms and values are dataframes containing augmented samples.
    - distances (list): List of distances between augmented samples and a reference dataset.
    - threshold (float): The distance threshold. Augmented samples with distances less than or equal to this threshold will be included.

    Returns:
    - filtered_samples (dict): A dictionary containing filtered augmented samples where keys correspond to the augmentation algorithms, and values are dataframes.
    """
    filtered_samples = {}

    # Iterate over the keys and values in the augmented_bucket dictionary
    for i, (key, synth_data) in enumerate(augmented_bucket.items()):
        # Check if the distance for the current sample is below or equal to the threshold
        if distances[i] <= threshold:
            # Include the augmented sample in the filtered_samples dictionary
            filtered_samples[key] = synth_data

    return filtered_samples


def split_dataset(data: pd.DataFrame, val_size=0.3, random_state=None):
 
    train_df, val_df = train_test_split(data, test_size=val_size,
                                        stratify=data["label"],
                                        random_state=random_state)
    return train_df, val_df


def main(data: pd.DataFrame, iterations: int):
    
    train, validation = split_dataset(data, 0.2, 42)
    augmented_df_final = pd.DataFrame()
    # Filetr validation for data similarity purposes
    most_common_label = validation['label'].mode()[0]
    filtered_validation = validation[validation['label'] != most_common_label]
 
    # Create bucketes
    distances_iteration_wass = []
 
    for iteration in range(iterations):

        buckets = bucketizer(train)
        augmented_buckets = []
        distances = []
        # Apply data augmentation for each bucket.
        for bucket in buckets:
            augmented_bucket = data_augmentation(bucket)

            # Calculate distances for each synth bucket

            for key, synth_data in augmented_bucket.items():
                dist = dataset_similarity(filtered_validation.drop("label", axis=1),
                                            synth_data.drop("label", axis=1))
                distances.append(dist)

            augmented_buckets.append(augmented_bucket)

        # Calculate the first quartile of distances
        threshold = np.percentile(distances, 25)
        print(f"[DEBUG]: Threshold {threshold}")

        filtered_augmented_samples = []
        # Filter the bucketes by distance
        for i, augmented_bucket in enumerate(augmented_buckets):
            filtered_samples = filter_augmented_samples(augmented_bucket,
                                                        distances[
                                                        i * len(augmented_bucket):(i + 1) * len(augmented_bucket)],
                                                        threshold)
            filtered_augmented_samples.append(filtered_samples)

        # Create augemnted_dataset by joining the different augmentations techinques.
        list_augmented = []
        for dict_augmented in filtered_augmented_samples:
            if not dict_augmented:
                continue
            else:
                for key, df in dict_augmented.items():
                    list_augmented.append(df)
                 
        augmented_df = pd.concat(list_augmented)
        augmented_df_final = pd.concat([augmented_df, augmented_df_final])
        distance_val_aug = dataset_similarity(filtered_validation.drop("label", axis=1),
                                            augmented_df_final.drop("label", axis=1))
        
        distances_iteration_wass.append(distance_val_aug)
    print(distances_iteration_wass)

    return distances_iteration_wass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer

    breast_cancer = load_breast_cancer()

    # Create a Pandas DataFrame
    data = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    data['label'] = breast_cancer.target
    data = data.rename(columns=lambda x: x.replace(' ', '_'))

    number_iterations = 100
    distances = main(data, number_iterations)
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances, marker='o')
    plt.title('Wasserstein Distances over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Wasserstein Distance')
    plt.grid(True)
    plt.show()