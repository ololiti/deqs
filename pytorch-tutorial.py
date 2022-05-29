import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import basemodel
import neuralodemodel
import deqmodel
import matplotlib.pyplot as plt
import time

num_epochs = 100

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

    times = [0]
    accuracy = [modeltype.test(test_dataloader, model, loss_fn)]
    start_time = time.time()
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        modeltype.train(train_dataloader, model, loss_fn, optimizer)
        curr_acc = modeltype.test(test_dataloader, model, loss_fn)
        end_time = time.time()
        times.append(end_time - start_time)
        accuracy.append(curr_acc)
        if times[len(times)-1] >= 600:
            break
    print("Done!")

    torch.save(model.state_dict(), pathname)
    print(f"Saved PyTorch Model State to {pathname}")

    return times, accuracy


def plot(base_accuracy, deq_accuracy):
    plt.figure()
    epochslist = [i for i in range(num_epochs+1)]
    times, accuracy = base_accuracy
    plt.plot(times, accuracy, 'xkcd:blurple', label='baseline')
    times, accuracy = deq_accuracy
    plt.plot(times, accuracy, 'xkcd:lavender', label='deq')

    plt.xlabel('time (s)')
    plt.ylabel('accuracy')
    plt.xlim(0, 650)
    plt.ylim(0, 100)
    plt.legend()

    plt.savefig("accuracy_plot.png")


base_accuracy = loadandtrain(basemodel, "basemodel.pth")
deq_accuracy = loadandtrain(deqmodel, "deqmodel.pth")
# ode_accuracy = loadandtrain(neuralodemodel, "odemodel.pth")


plot(base_accuracy, deq_accuracy)
