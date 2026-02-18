import torch 
import torch.nn.functional as f
import torch.nn as nn
from src.helper_function.matGen import genScaledRandMat
from src.helper_function.genKernel import gabor_bank, smooth_random_kernels
from src.helper_function.memory import get_model_param_size

class pretrainedCNN(nn.Module):

    # def __init__(self, number_of_kernels, input_dim, kernel_size, input_channels, output_channels, number_of_layers = 2, stride = 1, pad = 1, internal_channels = None):
    def __init__(self, kernel_size, input_channels, output_channels, number_of_layers = 2, stride = 1, pad = 1, internal_channels = None):

        super(pretrainedCNN, self).__init__()
        # self.number_of_kernels = number_of_kernels
        # self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.number_of_layers = number_of_layers
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.pad = pad
        self.internalChannels = internal_channels
        self.model = None
        self.initModel()
        self.genWeights()

    def initModel(self):
        if self.internalChannels is None:
            num_channels = [self.input_channels]
            for i in range(self.number_of_layers):
                if i == 0:
                    num_channels.append( self.input_channels + (self.output_channels - self.input_channels) // self.number_of_layers)

                if i == self.number_of_layers - 1:
                    num_channels.append(self.output_channels)

            print(num_channels)
            self.internalChannels = num_channels
            
        modelLayers = []
        for i in range(self.number_of_layers):
            modelLayers.append(nn.Conv2d(self.internalChannels[i], self.internalChannels[i+1], self.kernel_size, self.stride, self.pad))
            modelLayers.append(nn.MaxPool2d(2))

        self.model = nn.Sequential(*modelLayers)

    def getWeights(self):
        # for name, weights in self.model.named_parameters():
        #     print(weights)
        return self.model.named_parameters()

    def getStateDict(self):
        return self.model.state_dict()
    
    def genWeights(self):
        channels = self.internalChannels
        names, params = self.getWeights()
        weights = {}
        for n in names:
            n_id, n_name = n.split(".")
            i = int(n_id)
            if i == 0:
                currentWeights = gabor_bank(channels[i+1], channels[i], self.kernel_size)
            else:
                currentWeights = smooth_random_kernels(channels[i+1], channels[i], self.kernel_size)

            if n_name == "weight":
                weights[n] = currentWeights

        self.model.load_state_dict(weights)


    def forward(self, x):
        inputs = x
        for module in self.model:
            inputs = module(inputs)
        return inputs
    