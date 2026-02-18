import torch 
import torch.nn.functional as f
import torch.nn as nn
import numpy as np
import random
import math
from src.helper_function.matGen import genPCAMat, genScaledRandMat


class genMLP(nn.Module):
    def __init__(self, input_dimensions, output_dimensions, internal_dimensions=None, budget=None):
        super(genMLP, self).__init__()
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.internal_dimensions = internal_dimensions
        self.budget = budget
        self.model = self.genModel()

    def genModel(self):
        modelLayers = []
        if self.budget is None:
            num_dim = 2
            if self.internal_dimensions is None:
                self.internal_dimensions = [int((self.input_dimensions + self.output_dimensions)/2), self.output_dimensions]
            print(self.internal_dimensions)
            for i in range(num_dim):
                if i == num_dim:
                    modelLayers.append(nn.Linear(self.internal_dimensions[i], self.output_dimensions))
                elif i == 0:
                    modelLayers.append(nn.Linear(self.input_dimensions, self.internal_dimensions[i]))
                else:
                    modelLayers.append(nn.Linear(self.internal_dimensions[i-1], self.internal_dimensions[i]))
                modelLayers.append(nn.ReLU())
        else:
            num_dim = self.budget
            if self.internal_dimensions is None:
                self.internal_dimensions = [int(self.input_dimensions + self.output_dimesnions)/2, self.output_dimensions]

            for i in range(num_dim):
                if i == num_dim:
                    modelLayers.append(nn.Linear(self.internal_dimensions[i-1], self.output_dimensions))
                elif i == 0:
                    modelLayers.append(nn.Linear(self.input_dimensions, self.internal_dimensions[i]))
                else:
                    modelLayers.append(nn.Linear(self.internal_dimensions[i-1], self.internal_dimensions[i]))

                modelLayers.append(nn.ReLU())
        self.model = nn.Sequential(*modelLayers)


    def getStateDict(self):
        if self.model is not None:
            return self.model.state_dict()

    def genWeights(self):
        """
        Generate new weights
        """
        state_dict = self.getStateDict()
        new_state_dict = state_dict.copy()
        for name, thing in state_dict.items():
            if "weight" in name:
                nRows, nCols = thing.shape
                # print("shape of the matrix: ", nRows, nCols)
                tempMat = genScaledRandMat(nRows, nCols)
                new_state_dict[name] = tempMat.squeeze().T

        self.model.load_state_dict(new_state_dict)    

    def forward(self, X):
        y = self.model(X)
        return y
        
    def getEmbeddings(self, data):
        output = self.model(data)
        return output
