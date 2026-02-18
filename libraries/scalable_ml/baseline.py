import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score
from src.helper_function.memory import estimate_memory_training, estimate_memory_inference


from ucimlrepo import fetch_ucirepo 

try:
    results_csv = pd.read_csv("results.csv")
except OSError:
    results_csv = pd.DataFrame(data = 
        {"method":[],
         "accuracy" : [],
        "Recall" :[],
        "precision":[],
        "training":[],
        "model":[],
        "time": [],
        "memory":[]}
        )
    
print("result data: ", results_csv)

print("loading data set")
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 

headers = list(adult.variables["name"])
types = list(adult.variables["type"])
header_type_map = {}
for i in range(len(headers)):
    if "Cat" in types[i]:
        if types[i] not in header_type_map:
            header_type_map[types[i]] = [headers[i]]
        else:
            header_type_map[types[i]].append(headers[i]) 
    else:
        if "numerical" not in header_type_map:
            header_type_map["numerical"] = [headers[i]]
        else:
            header_type_map["numerical"].append(headers[i])

            cat_encoders = {}
for label in header_type_map:
    if "Cat" in label:
        headers = header_type_map[label]
        for head in headers:
            tempEncoder = OneHotEncoder()
            tempEncoder.fit(X[head].values.reshape(-1,1))
            cat_encoders[head] = tempEncoder

input_data = None
for head in (X.columns):
    if head in cat_encoders:
        temp_data = cat_encoders[head].transform(X[head].values.reshape(-1,1)).toarray()
        if input_data is None:
            input_data = temp_data
        else:
            input_data = np.concatenate((input_data, temp_data), axis=1)
        
        
temp_y = y.replace({"<=50K.":"<=50K", ">50K.":">50K"})
enc_y = LabelEncoder()
enc_y.fit(temp_y.values.reshape(-1,1))
encoded_y = enc_y.transform(temp_y.values.reshape(-1,1))

num_dp, num_input = input_data.shape
num_outputs = 52

train_index = list(np.random.choice(range(num_dp), size=int(num_dp*0.7)))
test_index = [index for index in range(num_dp) if index not in train_index]

raw_train_input = input_data[train_index, :]
raw_test_input = input_data[test_index, :]

train_label = encoded_y[train_index]
test_label = encoded_y[test_index]

#### start training for experiments
print("start experiment for autoencoder training")
modelStartTime = time.time()
from src.pretrainedAEs.autoencoder import AE
baseModel = AE(num_input, num_outputs)
baseModel.init_model()
baseModel.trainModel((torch.tensor(input_data, dtype=torch.float32), torch.tensor(input_data, dtype=torch.float32)),epochs=300, lr=0.05)

inter_train_input = baseModel.getEmbeddings(torch.tensor(raw_train_input, dtype=torch.float32)).cpu().detach().numpy()
inter_test_input = baseModel.getEmbeddings(torch.tensor(raw_test_input, dtype=torch.float32)).cpu().detach().numpy()

print("testing models")

testModel = xgb.XGBClassifier()
testModel.fit(inter_train_input, train_label)
pred_y = testModel.predict(inter_test_input)

modelEndTime = time.time()

estimateModelTrainMemory = estimate_memory_training(baseModel, torch.tensor(input_data, dtype=torch.float32), optimizer_type=torch.optim.SGD)
estimateModelInferenceMemory = estimate_memory_inference(baseModel, torch.tensor(input_data, dtype=torch.float32))

acc_score= accuracy_score(pred_y, test_label)
prec_score = precision_score(pred_y, test_label)
rec_score = recall_score(pred_y, test_label)

print("accuracy score: ", acc_score)
print("precision score: ", prec_score)
print("recall score: ", rec_score)

print("results collection", results_csv)

currentRes = {
    "method" : "trained AE + SGD + XGB",
    "accuracy" : acc_score,
    "Recall": rec_score, 
    "precision" : prec_score,
    "training": True,
    "model": "XGB",
    "time" : modelEndTime - modelStartTime,
    "memory": max(estimateModelTrainMemory, estimateModelInferenceMemory)
    }

results_csv.loc[len(results_csv)] = currentRes

print("experiment results ", results_csv)

startTime = time.time()
baselineClassifier = xgb.XGBClassifier()
baselineClassifier.fit(raw_train_input, train_label)
base_pred_y = baselineClassifier.predict(raw_test_input)
endTime = time.time()



raw_acc_score= accuracy_score(base_pred_y, test_label)
raw_prec_score = precision_score(base_pred_y, test_label)
raw_rec_score = recall_score(base_pred_y, test_label)

print("accuracy score: ", raw_acc_score)
print("precision score: ", raw_prec_score)
print("recall score: ", raw_rec_score)

currentRes = {
    "method" : "XGB",
    "accuracy" : raw_acc_score,
    "Recall":raw_rec_score, 
    "precision" : raw_prec_score,
    "training": False,
    "model": "XGB",
    "time" : endTime - startTime,
    "memory": 0
    }

results_csv.loc[len(results_csv)] = currentRes

from src.genAEs.genMLP import genMLP


print("Starting experiment with pretrained encoder")
pretrainStart = time.time()
PretrainedModel = genMLP(num_input, num_outputs)
PretrainedModel.genModel()
PretrainedModel.genWeights()

pretrained_inter_train = PretrainedModel(torch.tensor(raw_train_input, dtype=torch.float32)).detach().numpy()
pretrained_inter_test = PretrainedModel(torch.tensor(raw_test_input, dtype=torch.float32)).detach().numpy()

testModel = xgb.XGBClassifier()
testModel.fit(pretrained_inter_train, train_label)
pre_pred_y = testModel.predict(pretrained_inter_test)
pretrainEnd = time.time()

pre_acc_score= accuracy_score(pre_pred_y, test_label)
pre_prec_score = precision_score(pre_pred_y, test_label)
pre_rec_score = recall_score(pre_pred_y, test_label)

print("accuracy score: ", pre_acc_score)
print("precision score: ", pre_prec_score)
print("recall score: ", pre_rec_score)

generatedModelMemory = estimate_memory_inference(PretrainedModel, torch.tensor(input_data, dtype=torch.float32))

currentRes =  {
    "method" : "pretrained encoder + XGB",
    "accuracy" : pre_acc_score,
    "Recall": pre_rec_score, 
    "precision" : pre_prec_score,
    "training": False,
    "model": "XGB",
    "time" : pretrainEnd - pretrainStart,
    "memory": generatedModelMemory
    }


results_csv.loc[len(results_csv)] = currentRes

#$ sacve the dataframe
results_csv.to_csv("results.csv", index=False)