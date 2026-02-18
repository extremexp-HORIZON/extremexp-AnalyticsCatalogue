import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.genAEs.genCNNs import pretrainedCNN

transform = transforms.Compose([
    transforms.ToTensor(),  # convert image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # normalize with MNIST mean and std
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)



train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

imageModel = pretrainedCNN(3, 1, 20, 2)

imageModel.initModel()
weights = imageModel.getWeights()
print("weights:", weights)

for name, params in weights:
    if "weight" in name:
        print(name, params.shape)


