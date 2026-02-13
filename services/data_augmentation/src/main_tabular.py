import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer

import src.data_augmentation.ctgan_algorithm as da_ctgan
import src.data_augmentation.smote_algorithm as da_smote
import src.data_augmentation.adasyn_algorithm as da_adasyn
import src.metrics as metrics
import logging


def augment_data(data_path: str, label: str, model_type: str, n_samples: int, distance: str = "wassertein", verbose: bool = False) -> dict:
    """
    Apply data augmentation to the input dataset using the specified model.

    Parameters:
    - dataset (pd.DataFrame): The input dataset to be augmented.
    - label (str): The label or target variable for the augmentation.
    - model_type (str): The type of model used for data augmentation. Supported types are "CTGAN", "SMOTE", and "ADASYN".
    - n_samples (int): The number of synthetic samples to generate during augmentation.
    - verbose (bool, optional): If True, display additional information during augmentation. Default is False.

    Returns:
    - pd.DataFrame: Augmented dataset containing synthetic samples.

    Raises:
    - ValueError: If an invalid model type is provided or if data augmentation fails.
    """
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    if data_path is None:
        # For testing: load built-in breast cancer dataset
        data = load_breast_cancer()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
        dataset["target"] = data.target

    elif os.path.exists(data_path):
        # Load dataset from the provided path
        dataset = pd.read_pickle(data_path)

    else:
        # data_path is provided but the file doesn't exist
        raise ValueError(f"Dataset file not found at {data_path}")
      
    original_samples = dataset.shape[0]
    
    if model_type == "AUTO": 

        if distance not in ["pairwise", "wassertein"]:
            raise ValueError(f"Invalid distance type: {distance}") 
        
        augmented_data_ctgan = da_ctgan.ctgan_augmentation(dataset, label, n_samples)
        augmented_data_smote = da_smote.smote_augmentation(dataset, label, n_samples)
        augmented_data_adasyn = da_adasyn.adasyn_augmentation(dataset, label, n_samples)
         
        metric_smote = metrics.dataset_similarity(dataset.drop(label, axis=1), augmented_data_smote.drop(label, axis=1))
        metric_adasyn = metrics.dataset_similarity(dataset.drop(label, axis=1), augmented_data_adasyn.drop(label, axis=1))
        metric_ctgan = metrics.dataset_similarity(dataset.drop(label, axis=1), augmented_data_ctgan.drop(label, axis=1))
        
        logger.info(f"Applying DA using AUTO, Original Samples: {original_samples} - Generated Samples {n_samples}")
        logger.info(f"DA Quality Smote: {metric_smote}, Adasyn: {metric_adasyn} - CTGAN: {metric_adasyn}")

        # return the augmented data with more similarity to the original samples
        if distance == "pairwise":
            best_similarity_metric = max(abs(metric_smote[distance]), abs(metric_adasyn[distance]), abs(metric_ctgan[distance]))

        if distance == "wassertein":
            best_similarity_metric = min(metric_smote[distance], metric_adasyn[distance], metric_ctgan[distance])

        if best_similarity_metric == metric_smote[distance]: 
            return {"augmented_data": augmented_data_smote, 
                    "distances": metric_smote}
        
        elif best_similarity_metric == metric_adasyn[distance]: 
            return {"augmented_data": augmented_data_adasyn, 
                    "distances": metric_adasyn}
        else: 
            return {"augmented_data": augmented_data_ctgan, 
                    "distances": metric_ctgan}
        

    elif model_type in ["CTGAN", "SMOTE", "ADASYN"]: 
        augmentation_methods = {
            "CTGAN": da_ctgan.ctgan_augmentation,
            "SMOTE": da_smote.smote_augmentation,
            "ADASYN": da_adasyn.adasyn_augmentation
        }
    
        if model_type in augmentation_methods:
            logger.info(f"Applying DA using {model_type}, Original Samples: {original_samples} - Generated Samples {n_samples}")
            augmented_data = augmentation_methods[model_type](dataset, label, n_samples)
            print(augmented_data)

    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if augmented_data is None:
        logger.warning(f"Data augmentation failed for model type: {model_type}")
        raise ValueError(f"Data augmentation failed for model type: {model_type}")
    
    distances = metrics.dataset_similarity(dataset.drop(label, axis=1), augmented_data.drop(label, axis=1))
 
 
    return {"augmented_data": augmented_data, 
            "distances": distances}


