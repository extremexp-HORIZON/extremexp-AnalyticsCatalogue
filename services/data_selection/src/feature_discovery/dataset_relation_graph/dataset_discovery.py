import glob
import itertools
from typing import List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from valentine import valentine_match
from valentine.algorithms import Coma, Cupid

from src.feature_discovery.config import DATA_FOLDER, CONNECTIONS,SLASH
from src.feature_discovery.graph_processing.neo4j_transactions import merge_nodes_relation_tables


def profile_valentine_all(dataset:str , valentine_threshold: float = 0.55, files: List[str]=None, method="COMA", strategy="instance"):
    if files is None:
        files = glob.glob(f"{DATA_FOLDER}/{dataset}/*.csv", recursive=True)
        files = [f for f in files if CONNECTIONS not in f]
        print("files to look through ",files)

    profile_valentine_logic(files, valentine_threshold, method, strategy)


def profile_valentine_dataset(dataset_name: str, valentine_threshold: float = 0.55):
    files = glob.glob(f"{DATA_FOLDER / dataset_name}/*.csv", recursive=True)
    files = [f for f in files if CONNECTIONS not in f]

    profile_valentine_logic(files, valentine_threshold)


def profile_valentine_logic(files: List[str], valentine_threshold: float = 0.55, method="COMA", strategy="instance"):
    def profile(table_pair):
        print(table_pair)
        (tab1, tab2) = table_pair

        a_table_path = tab1.partition(f"{DATA_FOLDER}{SLASH}")[2]
        b_table_path = tab2.partition(f"{DATA_FOLDER}{SLASH}")[2]

        a_table_name = a_table_path.split(f"{SLASH}")[-1]
        b_table_name = b_table_path.split(f"{SLASH}")[-1]
        print(f"\nProcessing the match between:\n\t{a_table_path}\n\t{b_table_path}")

        # print("table 1 ", tab1)
        # print("table 2 ", tab2)
        df1 = pd.read_csv(tab1, encoding="utf8")
        df2 = pd.read_csv(tab2, encoding="utf8")

        if method == "COMA":
            if strategy == "instance":
                matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT_INST")) 
            else:
                matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT"))
        elif method == "Cupid":
            matches - valentine_match(df1, df2, Cupid())
        else:
            matches = valentine_match(df1, df2, Coma(strategy="COMA_OPT_INST")) 

        for item in matches.items():
            print(item)
            ((_, col_from), (_, col_to)), similarity = item
            if similarity > valentine_threshold:
                print(f"Similarity {similarity} between:\n\t{a_table_path} -- {col_from}\n\t{b_table_path} -- {col_to}")

                merge_nodes_relation_tables(a_table_name=a_table_name,
                                            b_table_name=b_table_name,
                                            a_table_path=a_table_path,
                                            b_table_path=b_table_path,
                                            a_col=col_from,
                                            b_col=col_to,
                                            weight=similarity)

                merge_nodes_relation_tables(a_table_name=b_table_name,
                                            b_table_name=a_table_name,
                                            a_table_path=b_table_path,
                                            b_table_path=a_table_path,
                                            a_col=col_to,
                                            b_col=col_from,
                                            weight=similarity)

    Parallel(n_jobs=-1)(delayed(profile)(table_pair) for table_pair in tqdm(itertools.combinations(files, r=2)))
