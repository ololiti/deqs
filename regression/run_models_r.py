import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import basemodel_r as basemodel
import matplotlib.pyplot as plt
import numpy as np
from generate_data_r import generate_test_data, generate_training_data

num_epochs = 30

def loadandtrain(modeltype, pathname, training_data, test_data):
    # Download training data from open datasets.
    # training_data = datasets.FashionMNIST(
    #     root="data",
    #     train=True,
    #     download=True,
    #     transform=ToTensor(),
    # )
    #
    # # Download test data from open datasets.
    # test_data = datasets.FashionMNIST(
    #     root="data",
    #     train=False,
    #     download=True,
    #     transform=ToTensor(),
    # )

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

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    accuracy = []
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        modeltype.train(train_dataloader, model, loss_fn, optimizer)
        curr_acc = modeltype.test(test_dataloader, model, loss_fn)
        accuracy.append(curr_acc)
    print("Done!")

    torch.save(model.state_dict(), pathname)
    print(f"Saved PyTorch Model State to {pathname}")

    return accuracy


def plot(base_accuracy, deq_accuracy=None):
    plt.figure()
    epochslist = [i+1 for i in range(num_epochs)]
    plt.plot(epochslist, base_accuracy, 'xkcd:blurple', label='baseline')
    if deq_accuracy is not None:
        plt.plot(epochslist, deq_accuracy, 'xkcd:lavender', label='deq')

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 100)
    plt.legend()

    plt.savefig("accuracy_plot.png")


#TODO: create a load-data-from-file option
test_data = generate_test_data()
print("Generated test data!")
training_data = generate_training_data()
print("Generated training data!")
print("Training baseline model...")
base_accuracy = loadandtrain(basemodel, "basemodel.pth", training_data, test_data)
# print("Training ODE model...")
# ode_accuracy = loadandtrain(neuralodemodel, "odemodel.pth", training_data, test_data)
# deq_accuracy = loadandtrain(deqmodel, "deqmodel.pth", training_data, test_data)

plot(base_accuracy)
