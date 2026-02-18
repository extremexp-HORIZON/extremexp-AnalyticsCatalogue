import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_iris
import src.data_augmentation.ctgan_algorithm as da_ctgan
import src.data_augmentation.smote_algorithm as da_smote
import src.data_augmentation.adasyn_algorithm as da_adasyn
import src.metrics as metrics
import statistics
import logging
from sklearn.model_selection import train_test_split



def split_dataset(data: pd.DataFrame, val_size=0.2, random_state=None):
    train_df, val_df = train_test_split(data, test_size=val_size,
                                        stratify=data["label"],
                                        random_state=random_state)
    return train_df, val_df

def augment_data(dataset: pd.DataFrame, label: str, model_type: str, n_samples: int, distance: str = "wassertein", verbose: bool = False) -> dict:
    train, validation = split_dataset(dataset, 0.2, None)
    n_samples = train.shape[0]
    most_common_label = validation['label'].mode()[0]
    filtered_validation = validation[validation['label'] != most_common_label]
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)
    
    augmentation_methods = {
        "CTGAN": da_ctgan.ctgan_augmentation,
        "SMOTE": da_smote.smote_augmentation,
        "ADASYN": da_adasyn.adasyn_augmentation
    }

    if model_type in augmentation_methods:
        logger.info(f"Applying DA using {model_type}, Original Samples: {train} - Generated Samples {n_samples}")
        augmented_data = augmentation_methods[model_type](train, label, n_samples)

    if augmented_data is None:
        logger.warning(f"Data augmentation failed for model type: {model_type}")
        raise ValueError(f"Data augmentation failed for model type: {model_type}")
    
    distances = metrics.dataset_similarity(filtered_validation.drop(label, axis=1), augmented_data.drop(label, axis=1))
    return {"augmented_data": augmented_data, 
            "distances": distances}

def evaluate_augmentations(data, label, model_types, n_samples, iterations=100):
    distances_dict_wass = {model: [] for model in model_types}
    distances_dict_pair = {model: [] for model in model_types}
    
    for iteration in range(iterations):
        print(f"Iteration {iteration}")
        for model in model_types:
            distance = augment_data(data, label, model, n_samples)
            distances_dict_wass[model].append(distance["distances"]["wassertein"])
            distances_dict_pair[model].append(distance["distances"]["pairwise"])
    
    for augmentation, distances in distances_dict_wass.items():
        print("wass-----------------------")
        mean_distance = statistics.mean(distances)
        var_distance = statistics.variance(distances)
        print(f"Augmentation type {augmentation}, MEAN: {mean_distance}, VAR: {var_distance}")

    for augmentation, distances in distances_dict_pair.items():
        print("pairwise-----------------------")
        mean_distance = statistics.mean(distances)
        var_distance = statistics.variance(distances)
        print(f"Augmentation type {augmentation}, MEAN: {mean_distance}, VAR: {var_distance}")


if __name__ == "__main__":
    
    datasets = {
        "Breast Cancer": load_breast_cancer,
        "Wine": load_wine,
        "Iris": load_iris}

    label = "label"
    model_types = ["SMOTE", "CTGAN", "ADASYN"]
    n_samples = 1000

    for dataset_name, dataset_loader in datasets.items():
        dataset = dataset_loader()
        data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
        data['label'] = dataset.target
        data = data.rename(columns=lambda x: x.replace(' ', '_'))

        print(f"Evaluating {dataset_name} Dataset")
        evaluate_augmentations(data, label, model_types, n_samples)
