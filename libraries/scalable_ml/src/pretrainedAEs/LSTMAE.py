import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from src.helper_function.matGen import genScaledRandMat


class LSTMAE(nn.Module):
    """
    LSTM autoencoder given input dimension and 
    """
    def __init__(self, input_dim, embedding_dim, num_lstm_layers=2, num_fc_layers = 0, batch_size = 1):
        super(LSTMAE, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_encodeing_layers = None
        self.num_decoding_layers = None
        self.num_fc_layers = 2
        self.encoderLSTM = nn.LSTM(self.input_dim, self.embedding_dim, num_layers = self.num_lstm_layers, batch_first=True)
        self.decoderLSTM = nn.LSTM(self.embedding_dim, self.input_dim, num_layers = self.num_lstm_layers, batch_first=True)
        self.dec_linear_layers = self.full_connected_decoding()
        self.enc_linear_layers = self.full_connected_encoding()
        self.batch_size = batch_size

        
    def full_connected_decoding(self):
        linear_layer_list = []
        for i in range(self.num_fc_layers):
            if i == self.num_fc_layers - 1:
                linear_layer_list.append(nn.Linear(self.embedding_dim , self.input_dim))
                linear_layer_list.append(nn.ReLU())

            elif i == 0:
                linear_layer_list.append(nn.Linear(self.input_dim, self.embedding_dim))
                linear_layer_list.append(nn.ReLU())

            else:
                linear_layer_list.append(nn.Linear(self.embedding_dim, self.embedding_dim))
                linear_layer_list.append(nn.ReLU())


        return nn.Sequential(*linear_layer_list)
    
    def full_connected_encoding(self):
        linear_layer_list = []
        for i in range(self.num_fc_layers):
            if i == 0:
                linear_layer_list.append(nn.Linear(self.embedding_dim, self.embedding_dim))
                linear_layer_list.append(nn.ReLU())
            else:
                linear_layer_list.append(nn.Linear(self.embedding_dim, self.embedding_dim))
                linear_layer_list.append(nn.ReLU())


        return nn.Sequential(*linear_layer_list)
    
    def forward(self, x):
        # print("input shape : ", x.shape)
        enc_h0 = torch.zeros(self.num_lstm_layers, self.batch_size, self.embedding_dim).to(x.device)
        enc_c0 = torch.zeros(self.num_lstm_layers, self.batch_size, self.embedding_dim).to(x.device)
        dec_h0 = torch.zeros(self.num_lstm_layers, self.batch_size , self.input_dim).to(x.device)
        dec_c0 = torch.zeros(self.num_lstm_layers, self.batch_size , self.input_dim).to(x.device)
        # print("h0 ", enc_h0.shape)

        enc_x , _ = self.encoderLSTM(x, (enc_c0, enc_h0))

        # print("encoded shape : ", enc_x.shape)

        dec_y , _ = self.decoderLSTM(enc_x, (dec_c0, dec_h0))

        y = self.dec_linear_layers(dec_y)
        return y

    def getEncodings(self, x, full_time_series = True):
        if isinstance(x, list):
            enc_h0 = torch.zeros(self.num_lstm_layers, x[0].size(0) , self.embedding_dim).to(x.device)
            enc_c0 = torch.zeros(self.num_lstm_layers, x[0].size(0) , self.embedding_dim).to(x.device)
        else:
            enc_h0 = torch.zeros(self.num_lstm_layers, x.size(0) , self.embedding_dim).to(x.device)
            enc_c0 = torch.zeros(self.num_lstm_layers, x.size(0) , self.embedding_dim).to(x.device)

        enc_x , _ = self.encoderLSTM(x, (enc_c0, enc_h0))
        if full_time_series:
            return enc_x
        else:
            return enc_x[:,-1, :]
        

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
                # print("generated matrix shape: ", tempMat.shape)
                new_state_dict[name] = tempMat.squeeze().T

        self.model.load_state_dict(new_state_dict) 
        