from pathlib import Path

import pandas as pd

from src.feature_discovery.config import DATA_FOLDER, ROOT_FOLDER, DATA
from src.feature_discovery.experiments.dataset_object import Dataset

CLASSIFICATION_DATASETS = set()
REGRESSION_DATASETS = set()
ALL_DATASETS = set()


def init_datasets():

    global CLASSIFICATION_DATASETS
    global REGRESSION_DATASETS
    global ALL_DATASETS

    CLASSIFICATION_DATASETS = set()
    REGRESSION_DATASETS = set()
    ALL_DATASETS = set()

    print("Initialising datasets ...")
    datasets_df = pd.read_csv(DATA + "/datasets.csv")

    for index, row in datasets_df.iterrows():
        dataset = Dataset(base_table_label=row["base_table_label"],
                          target_column=row["target_column"],
                          base_table_path=Path(row["base_table_path"]),
                          base_table_name=row["base_table_name"],
                          dataset_type=row["dataset_type"])
        if row["dataset_type"] == "regression":
            REGRESSION_DATASETS.add(dataset)
        else:
            CLASSIFICATION_DATASETS.add(dataset)

    # ALL_DATASETS.extend(CLASSIFICATION_DATASETS)
    # ALL_DATASETS.extend(REGRESSION_DATASETS)
    ALL_DATASETS = ALL_DATASETS.union(CLASSIFICATION_DATASETS)
    ALL_DATASETS = ALL_DATASETS.union(REGRESSION_DATASETS)
