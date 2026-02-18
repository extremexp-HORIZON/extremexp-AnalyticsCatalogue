from scipy.stats import wasserstein_distance
import pandas as pd 
import numpy as np
from scipy.stats import pearsonr


def compute_wassertein_distance(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:

   wassertein_distances = {col: wasserstein_distance(dataset_A[col], dataset_B[col]) for col in dataset_A.columns}
    # Calculate the mean Wasserstein distance
   mean_wassertein_distance = sum(wassertein_distances.values()) / len(wassertein_distances)

   return mean_wassertein_distance
   

def compute_pairwise_correlation(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> float:
   # Determine which dataset is smaller
   if len(dataset_A) <= len(dataset_B):
      smaller_dataset = dataset_A
      larger_dataset = dataset_B
   else:
      smaller_dataset = dataset_B
      larger_dataset = dataset_A

   # Randomly sample the larger dataset to match the length of the smaller one
   larger_dataset_sampled = larger_dataset.sample(n=len(smaller_dataset), replace=False, random_state=42)


   pairwise_correlation = {}

   for col_A in smaller_dataset.columns:
      # Check if the column exists in larger sampled dataset
      if col_A in larger_dataset_sampled.columns:
         # Compute Pearson correlation coefficient and p-value
         correlation_coefficient, _ = pearsonr(smaller_dataset[col_A], larger_dataset_sampled[col_A])
         # Store the correlation coefficient in the dictionary
         pairwise_correlation[col_A] = correlation_coefficient
      else:
         # Set NaN if the column doesn't exist in larger sampled dataset
         pairwise_correlation[col_A] = np.nan

   # Calculate the mean correlation coefficient
   mean_correlation = np.mean(list(pairwise_correlation.values()))

   return mean_correlation


def dataset_similarity(dataset_A: pd.DataFrame, dataset_B: pd.DataFrame) -> dict:

   """Calculates the Wasserstein distance-based similarity between two datasets.

    Parameters:
    - dataset_A (pd.DataFrame): The first dataset for comparison.
    - dataset_B (pd.DataFrame): The second dataset for comparison.

    Returns:
       - distances (dict):Dictionary contianing the different computed distances"""
 
   distances = {"wassertein": compute_wassertein_distance(dataset_A, dataset_B),
               "pairwise": compute_pairwise_correlation(dataset_A, dataset_B)}

   return distances


 