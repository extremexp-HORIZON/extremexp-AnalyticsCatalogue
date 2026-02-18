import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score
from src.helper_function.memory import estimate_memory_training, estimate_memory_inference
from src.genAEs.genRNN import pretrainedLSTM
from src.pretrainedAEs import LSTMAE
from tqdm import tqdm
from src.helper_function.data_transform import train_test_split
import torch.optim as optim
import torch.nn as nn

try:
    results_csv = pd.read_csv("lstm_results.csv")
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

def training(model, dataset, optimizer, criterion, epochs = 100):

    for i in tqdm(range(epochs)):
        if i % 20 == 0:
            agg_loss = 0
        for dp in dataset:
            optimizer.zero_grad()
            pred_y = model(dp)
            loss = criterion(pred_y, dp)
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                agg_loss += loss
        if i % 20 == 0:
            print("total agg loss: ", agg_loss)


print("loading dataset")


dataset = pd.read_csv("datasets/full_crypto_desktop.csv")
print(dataset.head())

train_data, test_data, counter = train_test_split(dataset, "user_id", "timestamp", "label", 0.2, "weighted")

_, input_dim = train_data[0][0].shape

train_data_tensor = []
train_label_tensor = []
test_data_tensor = []
test_label_tensor = []

for temp_input, temp_label in train_data:
    time_dim, input_dim = temp_input.shape
    train_data_tensor.append(torch.tensor(temp_input.values, dtype=torch.float32).reshape(1, time_dim, input_dim ))
    train_label_tensor.append(torch.tensor(temp_label.values, dtype=torch.float32))

for temp_input, temp_label in test_data:
    time_dim, input_dim = temp_input.shape
    test_data_tensor.append(torch.tensor(temp_input.values, dtype=torch.float32).reshape(1, time_dim, input_dim ))
    test_label_tensor.append(torch.tensor(temp_label.values, dtype=torch.float32))

print("training autoencoder experiment")
lstmStartTime = time.time()
encoder = LSTMAE(input_dim, input_dim//2)
optimizer = optim.SGD(encoder.parameters())
Criterion = nn.MSELoss()

training(encoder, train_data_tensor, optimizer, Criterion)

trained_ae_embeddings = encoder.getEncodings(train_data, False).detach().numpy()
trained_ae_test_embeddings = encoder.getEncodings(test_data, False).cpu().detach().numpy()

classifier  = xgb.XGBClassifier()
classifier.fit(trained_ae_embeddings, train_label_tensor)
pred_y = classifier.predict(trained_ae_test_embeddings)

lstm_ae_acc = accuracy_score(pred_y, test_label_tensor)
lstm_ae_prec = precision_score(pred_y, test_label_tensor)
lstm_ae_rec = recall_score(pred_y, test_label_tensor)

lstmEndTime = time.time()

results_csv.loc[len(results_csv)] = {
    "method": "Trained LSTMAE + GXB",
    "accuracy" : lstm_ae_acc,
    "Recall" : lstm_ae_rec,
    "precision": lstm_ae_prec,
    "training": True,
    "model": "LSTM + XGB",
    "time": lstmEndTime - lstmStartTime,
    "memory": 0
    }


results_csv.to_csv("lstm_results.csv", index=False)