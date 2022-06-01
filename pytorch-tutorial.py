import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import basemodel
import neuralodemodel
import matplotlib.pyplot as plt
import numpy as np
from generate_data import generate_test_data, generate_training_data
import deqmodel
import time

num_epochs = 40

def loadandtrain(modeltype, pathname, training_data, validation_data, test_data):
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
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in validation_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    print(f"Using {modeltype.device} device")

    model = modeltype.NeuralNetwork().to(modeltype.device)
    print(model)

    pos_weight = torch.from_numpy(np.array([0.66])).to(modeltype.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(modeltype.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    accuracy = []
    times = []
    start_time = time.time()
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        modeltype.train(train_dataloader, model, loss_fn, optimizer)
        curr_acc = modeltype.test(validation_dataloader, model, loss_fn)
        curr_time = time.time()
        accuracy.append(curr_acc)
        times.append(curr_time)
    print("Done!")

    torch.save(model.state_dict(), pathname)
    print(f"Saved PyTorch Model State to {pathname}")

    print("Checking the model on unseen test data:")
    modeltype.test(test_dataloader, model, loss_fn)

    return accuracy, times


def plot_epochs(base_accuracy, deq_accuracy):
    plt.figure()
    epochslist = [i+1 for i in range(num_epochs)]
    plt.plot(epochslist, base_accuracy, 'xkcd:blurple', label='baseline')
    plt.plot(epochslist, deq_accuracy, 'xkcd:lavender', label='deq')

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 100)
    plt.legend()

    plt.savefig("accuracy_plot.png")

def plot_time(base_accuracy, deq_accuracy):
    plt.figure()
    accuracy, times = base_accuracy
    plt.plot(times, accuracy, 'xkcd:blurple', label='baseline')
    accuracy, times = deq_accuracy
    plt.plot(times, accuracy, 'xkcd:lavender', label='deq')

    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    plt.xlim(0, times[len(times)-1])
    plt.ylim(0, 100)
    plt.legend()

    plt.savefig("accuracy_plot.png")


#TODO: create a load-data-from-file option
validation_data = generate_test_data()
print("Generated validation data!")
test_data = generate_test_data()
print("Generated test data!")
training_data = generate_training_data()
print("Generated training data!")
print("Training baseline model...")

base_accuracy = loadandtrain(basemodel, "basemodel.pth", training_data, validation_data, test_data)
# print("Training ODE model...")
# ode_accuracy = loadandtrain(neuralodemodel, "odemodel.pth", training_data, test_data)
deq_accuracy = loadandtrain(deqmodel, "deqmodel.pth", training_data, validation_data, test_data)

plot_time(base_accuracy, deq_accuracy)


