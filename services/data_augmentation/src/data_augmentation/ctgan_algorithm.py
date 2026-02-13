from ctgan import CTGAN
import pandas as pd


def ctgan_augmentation(df: pd.DataFrame, label: str, n_samples: int, random_state: int = None):
    """
    Apply CTGAN (Generative Adversarial Network for Tabular Data) augmentation to balance classes in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - label (str): The column name representing the target variable.
    - n_samples (int): The number of synthetic samples to generate for each minority class.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - pd.DataFrame: Augmented DataFrame with balanced classes.
    """

    targets = df[label].unique()
    common_label = df[label].mode().iloc[0]
    augmented_data = []
    for target in targets:
        if target == common_label:
            continue
 
        ctgan_model = CTGAN(epochs=100)
        ctgan_model.fit(df.loc[df[label] == target])
        #num_samples = len(df[df[label] == common_label]) - len(df[df[label] == target])
        synth_data = ctgan_model.sample(n_samples)
        synth_data[label] = target

        augmented_data.append(synth_data)

    return pd.concat(augmented_data)
