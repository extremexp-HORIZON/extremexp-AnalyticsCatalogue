import torch.nn as nn
import torch
import numpy as np
import sys


class taskEncoder(nn.Module):
    def __init__(self, dataType, memory = None, dataDim=None, outputDim = None, timeSeries = None):
        """ 
        Create an encoder based on the memory of the user and the datatype
        Inputs: taskType - tabular, time series
                memory - given a memory in megabytes
                dataDim - input dimensions of the data

        """
        super(taskEncoder, self).__init__()
        self.memoryBound = memory
        self.taskType = dataType
        self.dataDim = dataDim
        self.outputDim = outputDim
        self.timeSeries = timeSeries
        self.createModel()
    

    def dimensions(self):
        if self.taskType == "tabular" or "timeSeries":
            if self.dataDim > 10:

                if self.memoryBound is None:
                    if self.dataDim is not None:
                        if self.outputDim is None:
                            outputDim = self.dataDim //2
                            self.outputDim = outputDim
                    else:
                        raise ValueError("Data needs to be defined for the model")
                    
                    memoryBound = outputDim * self.dataDim * sys.getsizeof(float(2))
                    self.memoryBound = memoryBound
                    return [self.dataDim, self.outputDim]
                
                else:
                    if self.dataDim is not None:
                        if self.outputDim is None:
                            self.outputDim = self.dataDim // 2
                            return [self.dataDim, self.outputDim]

                        else:
                            if self.memoryBound > self.outputDim * self.dataDim * sys.getsizeof(float(2)):
                                # approx the memory of the weights
                                return [self.dataDim, self.outputDim]
                            else:
                                tempOutputDim = self.memoryBound // (self.dataDim * sys.getsizeof(float(2)))

                                if tempOutputDim > self.dataDim :
                                    # more memory ==> deeper model
                                    finalOutputDim = self.dataDim // 2


                                self.outputDim - finalOutputDim
                                return [self.dataDim, self.outputDim]
                    else:
                        raise ValueError("Data needs to be defined for the model")
                
        
            else:
                self.outputDim = self.dataDim 
                return [self.dataDim, self.outputDim]
        elif self.taskType == "image":
            # require a model that has more input parameters
            if self.memoryBound is None:
                inputDims = self.dataDim
                outputDims = self.dataDim * 3
                self.outputDim = outputDims
                return [inputDims, outputDims]
            
            else:
                if self.kernelSize is None:
                    kernelSize = 5
                else:
                    kernelSize = self.kernelSize
                    # checkl the kernel size
                if sys.getsizeof(float(2)) * kernelSize*kernelSize * self.inputDims <= self.memoryBound :
                    return {"kernelSize" : kernelSize, "inputChannels" : self.inputDims, "outputChannels" : outputDims}
                else:
                    numOutputChannels = self.memoryBound // inputDims * kernelSize * kernelSize
                    return {"kernelSize" : kernelSize, "inputChannels" : self.inputDims, "outputChannels" : numOutputChannels}
 
            
    def createModel(self):
        if self.taskType is not None:
            match self.taskType:
                case ("tabular"):
                    import src.genAEs.genMLP as MLP
                    dimensions = self.dimensions()
                    inputDim = dimensions[0]
                    outputDim = dimensions[-1]
                    
                    model = MLP.genMLP(input_dimensions=inputDim, output_dimensions=outputDim, internal_dimensions=dimensions)
                    model.genModel()
                    model.genWeights()

                    self.model = model

                case ("timeSeries") :
                    import src.genAEs.genRNN as RNN
                    dimensions = self.dimensions()
                    inputDims = dimensions[0]
                    outputDims = dimensions[-1]
                    model = RNN.pretrainedLSTM(input_dim=inputDims, output_dim=outputDims, num_layers=1)
                    model.genWeights()

                    self.model = model

                case ("TimeSeries") :
                    import src.genAEs.genRNN as RNN
                    dimensions = self.dimensions()
                    inputDims = dimensions[0]
                    outputDims = dimensions[-1]
                    model = RNN.pretrainedLSTM(input_dim=inputDims, output_dim=outputDims, num_layers=1)
                    model.genWeights()

                    self.model = model

                case ("Images"):
                    import src.genAEs.genCNNs as CNN
                    dimensions = self.dimensions()
                    inputChannels = dimensions["inputChannels"]
                    kernelSize = dimensions["kernelSize"]
                    outputChannels = dimensions["outputChannels"]
                    model = CNN.pretrainedCNN(input_channels=inputChannels, output_channels=outputChannels, kernel_size=kernelSize, number_of_layers=1)
                    model.genWeights()
                    self.model = model

        else:
            raise ValueError("task Type needs to be defined")

    def forward(self, x):
        return self.model.getEmbeddings(x)
    
    def getEmbeddings(self, x):
        match self.taskType:
            case ("tabular") :
                return self.model.getEmbeddings(x)
        
            case ("timeSeries"):
                
                if self.timeSeries is None:
                    return self.model(x, False)
                else:
                    return self.model(x, self.timeSeries)
            
            case ("image") :
                return self.model.getEmbeddings(x)
