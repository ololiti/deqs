import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import basemodel
import neuralodemodel

def loadandtrain(modeltype, pathname):
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    print(f"Using {modeltype.device} device")

    model = modeltype.NeuralNetwork().to(modeltype.device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        modeltype.train(train_dataloader, model, loss_fn, optimizer)
        modeltype.test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), pathname)
    print(f"Saved PyTorch Model State to {pathname}")

loadandtrain(basemodel, "basemodel.pth")
loadandtrain(neuralodemodel, "odemodel.pth")
