import torch 
import torch.nn.functional as f
import torch.nn as nn
from src.helper_function.matGen import genScaledRandMat


class pretrainedLSTM(nn.Module):
    def __init__(self, input_dim, num_layers, output_dim, internal_dims = None):
        super(pretrainedLSTM, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.internal_dims = internal_dims
        self.init_model()

    def init_model(self):
        if self.internal_dims is None:
            temp_dims = []
            for i in range(self.num_layers):
                if i == 0:
                    temp_dims.append(self.input_dim)
                if i == self.num_layers - 1:
                    temp_dims.append(self.output_dim)
                else:
                    temp_dims.append(int(self.input_dim + i*(self.input_dim + self.output_dim)/self.num_layers))
            
            self.internal_dims = temp_dims
        modelLayers = []
        for i in range(len(temp_dims)-1):
            modelLayers.append(nn.LSTM(self.internal_dims[i], self.internal_dims[i+1], batch_first=True))

        self.model = nn.Sequential(*modelLayers)

    def forward(self, x, full_series = True):
        layerInputs = x
        for i, module in enumerate(self.model):
            # print(module.shape)
            enc_h0 = torch.zeros(1, layerInputs.size()[0], self.internal_dims[i+1]).to(x.device)
            enc_c0 = torch.zeros(1, layerInputs.size()[0], self.internal_dims[i+1]).to(x.device)
            layerInputs, states = module(layerInputs, (enc_h0, enc_c0))

        if not full_series:
            return layerInputs[:, -1, :]
        else:
            return layerInputs
        
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
                tempMat = genScaledRandMat(nRows, nCols)
                print("generated matrix shape: ", tempMat.shape)
                new_state_dict[name] = tempMat.squeeze()

        self.model.load_state_dict(new_state_dict)    

    def getWeights(self):
        for name, module in self.model.named_parameters():
            print(module)
        

    def getEmbeddings(self, x, timeSeries):
        return 