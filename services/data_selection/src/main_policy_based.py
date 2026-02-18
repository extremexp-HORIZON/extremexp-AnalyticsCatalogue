import re
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance

import src.utils.utils as ut
import src.data_augmentation.ctgan_algorithm as da_ctgan
import src.data_augmentation.smote_algorithm as da_smote
import src.data_augmentation.adasyn_algorithm as da_adasyn


def print_debug_info(title, data, constraints):
    print("++++++++++++++++++++++++++++")
    print(f"[DEBUG]: {title}")
    constrained_data = constrainer(data, constraints)
    print("[DEBUG]: Labels original ")
    print(data["label"].value_counts())
    print("[DEBUG]: Labels constrained")
    print(constrained_data["label"].value_counts())
    print("++++++++++++++++++++++++++++")


def data_sanity(data: pd.DataFrame, constraints: list):
    """
    Perform data sanity checks to ensure the input DataFrame and constraints are valid.

    Parameters:
    - data (pd.DataFrame): Input DataFrame to be checked.
    - constraints (list): List of constraints to be applied to the data.

    Raises:
    - ValueError: If the input DataFrame is empty.
    - ValueError: If the constraints list is empty.
    - ValueError: If columns mentioned in constraints do not exist in the dataset.
    """

    if data.empty:
        raise ValueError("Input DataFrame is empty")
    print(f"[DEBUG]: Input Dataset shape: {data.shape}")

    if not constraints:
        raise ValueError("Constraints list is empty")

    print(f"[DEBUG]: Input constraints len: {len(constraints)}")

    # Check if columns mentioned in constraints exist in the dataset
    columns_in_constraints = [re.split(r'(<|>|==)', constraint)[0].strip() for constraint in constraints]
    invalid_columns = [col for col in columns_in_constraints if col not in data.columns]
    if invalid_columns:
        raise ValueError(f"The following columns do not exist in the dataset: {', '.join(invalid_columns)}")

    print(f"[DEBUG]: Applying constraints: {constraints}")


def constrainer(dataset: pd.DataFrame, constraints: list) -> pd.DataFrame:
    """
    Apply constraints to filter a DataFrame.

    Parameters:
    - dataset (pd.DataFrame): Input DataFrame to be filtered.
    - constraints (list): List of constraints to be applied using the 'query' method.

    Raises:
    - ValueError: If there is an error parsing the constraints using the 'query' method.
    - ValueError: If no rows match the specified constraints.

    Returns:
    - pd.DataFrame: DataFrame containing rows that satisfy the specified constraints.
    """

    # Combine constraints into a single string and filter the dataset
    combined_constraint = " and ".join(constraints)
    try:
        dataset_constrained = dataset.query(combined_constraint)
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing constraints: {e}")

    if dataset_constrained.empty:
        raise ValueError("No rows match the specified constraints")

    print(f"[DEBUG]: Original shape: {dataset.shape}")
    print(f"[DEBUG]: Contrained shape: {dataset_constrained.shape}")
    return dataset_constrained


def bucketizer(dataset: pd.DataFrame) -> list:
    """
    Creates a list of subsets (buckets) from the given dataset for data augmentation purposes.

    Parameters:
    - dataset (pd.DataFrame): The input dataset to be divided into subsets.

    Returns:
    - subsets (list): A list of subsets (buckets), each containing a fixed number of samples for data augmentation.
    """

    N = len(dataset)
    # Set the number of subsets (M) and size of each subset (L)
    # M = int(0.10 * N)
    # L = int(N / M)*10
    M = 5
    L = 100

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
    print("------------------------------------------------------------------------")
    print(dataset["label"].value_counts())
    synth_data = {}
    synth_data["SMOTE"] = da_smote.smote_augmentation(dataset)
    synth_data["ADASYN"] = da_adasyn.adasyn_augmentation(dataset)
    # synth_data["CTGAN"] = da_ctgan(dataset)

    for key, value in synth_data.items():
        print(f"[DEBUG]: ALGORITHM: {key}, SHAPE: {value.shape}")
        print(value["label"].value_counts())

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


def main(data: pd.DataFrame, constraints: list):
    try:
        data_sanity(data, constraints)

    except ValueError as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}

    train, validation = ut.split_dataset(data, random_state=ut.SEED)
    # Apply constrains
    try:
        print_debug_info("Train dataset", train, constraints)
        train_constrained = constrainer(train, constraints)

        print_debug_info("Validation dataset", validation, constraints)
        validation_constrained = constrainer(validation, constraints)

    except ValueError as e:
        return {"error": str(e)}

    # Create bucketes
    buckets = bucketizer(train_constrained)
    augmented_buckets = []
    distances = []
    # Apply data augmentation for each bucket.
    for bucket in buckets:
        augmented_bucket = data_augmentation(bucket)

        # Calculate distances for each synth bucket

        for key, synth_data in augmented_bucket.items():
            dist = dataset_similarity(validation_constrained.drop("label", axis=1),
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

    list_augmented = []
    for dict_augmented in filtered_augmented_samples:
        if not dict_augmented:
            continue
        else:
            for key, df in dict_augmented.items():
                list_augmented.append(df)

    augmented_df = pd.concat(list_augmented)
    final_dataset = pd.concat([train, augmented_df])
    print(train.shape)
    print(augmented_df.shape)
    print(final_dataset.shape)
    print(final_dataset["label"].value_counts())

    return True


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    breast_cancer = load_breast_cancer()

    # Create a Pandas DataFrame
    data = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    data['label'] = breast_cancer.target
    data = data.rename(columns=lambda x: x.replace(' ', '_'))
    constraints = ["mean_radius<20", "mean_compactness<0.1"]
    main(data, constraints)