import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels = 6, pool_kernel = 2):
        """
        input_dim = number of features, size of the kernel
        """
        super().__init__()
        self.num_feats =input_dim
        self.num_channels = num_channels
        self.pool_kernel = 2
        self.conv1 = nn.Conv2d(1, self.num_channels, input_dim, padding=int(input_dim/2))
        self.pool = nn.MaxPool1d(self.pool_kernel)
        self.fc1 = nn.Linear(6 * 142*83, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_dim)

    def flatten(self, x):
        if len(x.shape) == 4:
            return torch.flatten(x,1)
        elif len(x.shape) == 3:
            return torch.flatten(x, 0)
        else:
            raise NotImplementedError("Wrong dimensions")

    def forward(self, x):
        print(x.shape)
        pad = (int(self.num_feats/2), int(self.num_feats/2), 0, 0)  # Pad 2 pixels on the left and right, 0 on top and bottom
        x = F.pad(x, pad, mode='constant', value=0)  # Zero paddin
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


def fit(model, train_loader, optimizer = "SGD", loss=None, num_epochs = 1000, lr=0.001, momentum=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    if loss is None:
        loss_func = nn.CrossEntropyLoss
    if optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr)
    
        
    num_epochs = 1000
    running_loss = 0
    loss_vals = []

    for e in range(num_epochs):
        for input_vals, labels in train_loader:
            optimizer.zero_grad()
            input_vals = input_vals.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            outputs = model(input_vals[:, None, :, :]).to(device)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if e % 100 == 99:
                running_loss += loss.item()
                
        if e % 100 == 99:
            loss_vals.append(running_loss)
            print(f"Loss at current epocch {e} : {running_loss}")
            running_loss = 0

    print('Finished Training')  
    return model


def test_model(model, test_data, test_labels):
    res = []
    for dp in test_data:
        res = model(test_data)