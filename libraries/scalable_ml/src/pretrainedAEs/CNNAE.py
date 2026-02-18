import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim


class CNNAE(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride = 1, padding = 1, num_layers = 1, internal_channels = None):
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.num_layers = num_layers
        self.internal_channels = internal_channels

    def init_model(self):
        if self.internal_channels is None:
