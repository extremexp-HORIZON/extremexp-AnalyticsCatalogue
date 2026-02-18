
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

def train_test_split(dataframe, splitIndex, timeIndex, labelIndex, proportion, sampleMethod = "random"):
    index_set  = list(set(dataframe[splitIndex].values))
    labels = list(set(dataframe[labelIndex].values))
    class_label = {}
    for label in labels:
        class_label[label] = []

    for idx in index_set:
        class_label[dataframe[labelIndex][dataframe[splitIndex] == idx].unique()[0]].append(idx)

    train_label = []
    for labelClass in class_label:
        train_label += [labelClass] * len(class_label[labelClass])
    

    if sampleMethod == "random":
        test_id = list(np.random.choice(index_set, size=int(len(index_set)*proportion), replace=False))
        train_id = [index for index in index_set if index not in test_id]
        sampled_train_data = []
        for index in train_id:
            temp_data = dataframe[dataframe[splitIndex] == index].reset_index()
            temp_label = temp_data[labelIndex]
            temp_data = temp_data.drop([labelIndex], axis=1)
            temp_data = temp_data.sort_values(by = timeIndex)
            temp_data = temp_data.drop([timeIndex, splitIndex], axis=1)
            sampled_train_data.append((temp_data, temp_label))

        sampled_test_data = []
        for index in test_id:
            temp_data = dataframe[dataframe[splitIndex] == index].reset_index()
            temp_label = temp_data[labelIndex]
            temp_data = temp_data.drop([labelIndex], axis=1)
            temp_data = temp_data.sort_values(by = timeIndex)
            temp_data = temp_data.drop([timeIndex, splitIndex], axis=1)
            sampled_test_data.append((temp_data, temp_label))

        return sampled_train_data, sampled_test_data, _


    elif sampleMethod == "weighted":
        class_weights = compute_class_weight(
                                            class_weight = "balanced",
                                            classes = np.unique(train_label),
                                            y = train_label                                                    
                                        )
        class_weights = dict(zip(np.unique(train_label), class_weights))
        class_weights_tensor = torch.tensor(list(class_weights.values()))
        test_id = []
        train_id = []
        test_class_samples = {}
        train_class_samples = {}
        for classes in class_label:
            select_indices = np.random.choice(class_label[classes], size=int(len(class_label[classes])*proportion), replace=False)
            test_class_samples[classes] = select_indices
            test_id += list(select_indices)

        for classes in class_label:
            temp_user_ids = [us_id for us_id in class_label[classes] if us_id not in test_class_samples[classes]]
            train_class_samples[classes] = temp_user_ids
            train_id += temp_user_ids
        
        sampled_train_data = []
        for index in train_id:
            temp_data = dataframe[dataframe[splitIndex] == index].reset_index()
            temp_label = temp_data[labelIndex]
            temp_data = temp_data.drop([labelIndex], axis=1)
            temp_data = temp_data.sort_values(by = timeIndex)
            temp_data = temp_data.drop([timeIndex, splitIndex], axis=1)
            sampled_train_data.append((temp_data, temp_label))

        sampled_test_data = []
        for index in test_id:
            temp_data = dataframe[dataframe[splitIndex] == index].reset_index()
            temp_label = temp_data[labelIndex]
            temp_data = temp_data.drop([labelIndex], axis=1)
            temp_data = temp_data.sort_values(by = timeIndex)
            temp_data = temp_data.drop([timeIndex, splitIndex], axis=1)
            sampled_test_data.append((temp_data, temp_label))

        return sampled_train_data, sampled_test_data, class_weights_tensor
