from imblearn.over_sampling import ADASYN
import pandas as pd


def adasyn_augmentation(df: pd.DataFrame, label: str, n_samples: int, random_state: int = None):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to augment a DataFrame and balance classes.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - label (str): The column name representing the target variable.
    - n_samples (int): The desired number of synthetic samples to generate.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - pd.DataFrame: Augmented samples.
    """
   
    x = df.drop(label, axis=1)
    y = df[label]

    # Calculate the desired number of synthetic samples for the minority class
    minority_class = df[label].value_counts().idxmin()
    majority_class = df[label].value_counts().idxmax()
    existing_minority_samples = df[label].value_counts().min()
    existing_majority_samples = df[label].value_counts().max()
    desired_minority_samples = min(n_samples, existing_minority_samples)

    # Apply ADASYN
 
    sampling_strategy = {majority_class: existing_majority_samples, minority_class: existing_minority_samples + desired_minority_samples}
 
    adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
 
    X_resampled, y_resampled = adasyn.fit_resample(x, y)
 
    # Create a DataFrame for the resampled data
    synth_df = pd.DataFrame(data=X_resampled, columns=x.columns)
    synth_df[label] = y_resampled

    # Take only synthetic samples by dropping real ones
    synth_df = synth_df[~synth_df.isin(df.to_dict(orient='list')).all(axis=1)]

    return synth_df

