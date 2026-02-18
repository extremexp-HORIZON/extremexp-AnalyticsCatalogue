import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from autoencoder import mlp

class AE(nn.Module):
    def __init__(self, input_dim, output_dim, internal_dims=None, budget=None, optimizer = None):
        super(AE, self).__init__()
        self.budget = budget
        self.input_dimensions = input_dim
        self.output_dimensions = output_dim
        self.internal_dimensions = internal_dims
        self.encoder = None
        self.decoder = None
    
    
    def init_model(self):
        self.encoder = mlp(self.input_dimensions, self.output_dimensions, self.internal_dimensions, self.budget)
        self.encoder.initModel()
        if self.internal_dimensions is None:
            self.decoder = mlp(self.output_dimensions, self.input_dimensions, self.internal_dimensions, self.budget)
        else:
            self.decoder = mlp(self.output_dimensions, self.input_dimensions, reversed(self.internal_dimensions), self.budget)
        self.decoder.initModel()
        print("sanity check:" ,self.decoder)

    
    def forward(self, x):
        y = self.encoder(x)
        return y
    
    
    def trainModel(self, train_data, epochs=100, lr=0.05):
        from tqdm import tqdm
        optimizer = optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        lossFunction = nn.MSELoss()
        print("start training model")
        for i in tqdm(range(epochs)):
            optimizer.zero_grad()
            if i % 20 == 0:
                acc_loss = 0

            for dp in train_data:
                y_pred = self.decoder(self.encoder((dp)))
                loss = lossFunction(y_pred, dp)
                loss.backward()
                optimizer.step()
                if i % 20 == 0:
                    acc_loss += loss.cpu().detach()

            if i % 20 == 0:
                print(f"The acc loss at epcoh {i}: ", acc_loss)
        print("finished training model")


    def getWeights(self):
        for thing in self.encoder.parameters():
            print(thing)


    def getEmbeddings(self, testData):
        return self.encoder(testData)
    