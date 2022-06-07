import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import basemodel
import neuralodemodel
import naivemodel
import repeatedmodel
import matplotlib.pyplot as plt
import numpy as np
from generate_data import generate_test_data, generate_training_data
import deqmodel
import time
import csv

num_epochs = 40
np.random.seed(0)

def loadandtrain(modeltype, pathname, training_data, validation_data, test_data, multilayer=False):
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

    if multilayer:
        model = modeltype.NeuralNetwork(num_layers=3).to(modeltype.device)
    else:
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
        times.append(curr_time - start_time)
    print("Done!")

    torch.save(model.state_dict(), pathname)
    print(f"Saved PyTorch Model State to {pathname}")

    print("Checking the model on unseen test data:")
    modeltype.test(test_dataloader, model, loss_fn)

    return accuracy, times


def plot_epochs(accuracies, names):
    COLORS = ['xkcd:blurple', 'xkcd:lavender', 'xkcd:lightblue', 'xkcd:indigo', 'xkcd:pink']
    plt.figure()
    epochslist = [i+1 for i in range(num_epochs)]
    for i in range(len(accuracies)):
        plt.plot(epochslist, accuracies[i][0], COLORS[i], label=names[i])

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 100)
    plt.grid()
    plt.legend()

    plt.savefig("accuracy_epoch_plot.png")

def plot_time(accuracies, names):
    COLORS = ['xkcd:blurple', 'xkcd:lavender', 'xkcd:lightblue', 'xkcd:indigo', 'xkcd:pink']
    plt.figure()
    for i in range(len(accuracies)):
        plt.plot(accuracies[i][1], accuracies[i][0], COLORS[i], label=names[i])

    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    times = accuracies[len(accuracies)-1][1]
    plt.xlim(0, times[len(times)-1])
    plt.ylim(0, 100)
    plt.legend()

    plt.savefig("accuracy_time_plot.png")


mycsv = open("accuracies.csv", "w")
writer = csv.writer(mycsv)
#TODO: create a load-data-from-file option
validation_data = generate_test_data()
print("Generated validation data!")
test_data = generate_test_data()
print("Generated test data!")
training_data = generate_training_data()
print("Generated training data!")
print("Training baseline model...")

naive_accuracy = loadandtrain(naivemodel, "naivemodel_exp2.pth", training_data, validation_data, test_data)
base_accuracy = loadandtrain(basemodel, "basemodel_exp2.pth", training_data, validation_data, test_data)
multilayer_accuaracy = loadandtrain(basemodel, "mlmodel_exp2.pth", training_data, validation_data, test_data, multilayer=True)
repeated_accuracy = loadandtrain(repeatedmodel, "repeatedmodel_exp2.pth", training_data, validation_data, test_data)
deq_accuracy = loadandtrain(deqmodel, "deqmodel_exp2.pth", training_data, validation_data, test_data)


accuracies = [naive_accuracy, base_accuracy, multilayer_accuaracy, repeated_accuracy, deq_accuracy]
for accuracy in accuracies:
    writer.writerow(accuracy[0])
names = ["basic NN", "GRU (1 layer)", "GRU (3 layer)",  "GRU (repeated layer)", "DEQ"]
plot_epochs(accuracies, names)
plot_time(accuracies, names)

mycsv.close()



