import pandas as pd
from scipy.io import arff
import numpy as np
from src.pretrainedAEs.autoencoder import LSTMAE
from src.genAEs.genRNN import pretrainedLSTM
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score


trainData = arff.loadarff("datasets/atrialFibrillation/AtrialFibrillation_Train.arff")
testData = arff.loadarff("datasets/atrialFibrillation/AtrialFibrillation_Test.arff")
trainDF = pd.DataFrame(trainData[0])
testDF = pd.DataFrame(testData[0])

print(trainDF.head())
train_input = trainDF["ECG_Atrial_Fibrilation"]
train_label = trainDF["target"]

test_input = testDF["ECG_Atrial_Fibrilation"]
test_label = testDF["target"]

train_data_tensor = []
for thing in train_input:

    temp_array = np.array(thing.tolist())
    dim1, dim2 = temp_array.shape
    train_data_tensor.append(torch.tensor(temp_array, dtype=torch.float32).reshape(1, dim2, dim1))

train_data_tensor = torch.stack(train_data_tensor, dim=0)

print(train_data_tensor.shape)


labeler = LabelEncoder()
labeler.fit(train_label)

transformed_train = labeler.transform(train_label)
transformed_test = labeler.transform(test_label)

test_data_tensor = []
for thing in test_input:
    temp_array = np.array(thing.tolist())
    dim1, dim2 = temp_array.shape
    test_data_tensor.append(torch.tensor(temp_array, dtype=torch.float32).reshape(1, dim2, dim1))

test_data_tensor = torch.stack(test_data_tensor, dim=0)


lstmAE = LSTMAE(2, 2)

optimizer = optim.SGD(lstmAE.parameters())
critereon = nn.MSELoss()

for i in tqdm(range(100)):
    if i % 20 == 0:
        agg_loss = 0
    optimizer.zero_grad()
    for dp in train_data_tensor:
        pred_y = lstmAE(dp)
        loss = critereon(pred_y, dp)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            agg_loss += loss

    if i % 20 == 0:
        print("agg loss: ", agg_loss)


intermediateTrain = lstmAE.getEncodings(train_data_tensor.squeeze(), False).cpu().detach().numpy()
intermediateTest = lstmAE.getEncodings(test_data_tensor.squeeze(), False).cpu().detach().numpy()


testingModel = xgb.XGBClassifier()

testingModel.fit(intermediateTrain, transformed_train)
pred_y = testingModel.predict(intermediateTest)

lstm_ae_acc = accuracy_score(pred_y, transformed_test)
lstm_ae_prec = precision_score(pred_y, transformed_test, average="micro")
lstm_ae_rec = recall_score(pred_y, transformed_test, average="micro")

print("accuracy score: ", lstm_ae_acc)
print("precision score: ", lstm_ae_prec)
print("recall score: ", lstm_ae_rec)


transferLSTM = pretrainedLSTM(2, 2, 2)
transferLSTM.init_model()
transferLSTM.genWeights()
pretrained_train_embeddings = transferLSTM(train_data_tensor.squeeze(), full_series=False).cpu().detach().numpy()
pretrained_test_embeddings = transferLSTM(test_data_tensor.squeeze(), full_series=False).cpu().detach().numpy()

testModel = xgb.XGBClassifier()
testModel.fit(pretrained_train_embeddings, transformed_train)
pred_y = testModel.predict(pretrained_test_embeddings)

pretrainedlstm_acc = accuracy_score(pred_y, transformed_test)
pretrainedlstm_prec = precision_score(pred_y, transformed_test, average="micro")
pretrainedlstm_rec = recall_score(pred_y, transformed_test, average="micro")

print("accuracy score: ", lstm_ae_acc)
print("precision score: ", lstm_ae_prec)
print("recall score: ", lstm_ae_rec)

