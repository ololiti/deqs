import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import basemodel
import neuralodemodel


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

print(f"Using {basemodel.device} device")
print(f"Using {neuralodemodel.device} device")

model = basemodel.NeuralNetwork().to(basemodel.device)
print(model)

odemodel = neuralodemodel.NeuralNetwork().to(neuralodemodel.device)
print(odemodel)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

ode_loss_fn = nn.CrossEntropyLoss()
ode_optimizer = torch.optim.SGD(odemodel.parameters(), lr=1e-3)

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     basemodel.train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn, basemodel.device)
# print("Done!")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    neuralodemodel.train(train_dataloader, odemodel, ode_loss_fn, ode_optimizer)
    test(test_dataloader, odemodel, ode_loss_fn, neuralodemodel.device)
print("Done!")


# torch.save(model.state_dict(), "basemodel.pth")
# print("Saved PyTorch Model State to basemodel.pth")

torch.save(odemodel.state_dict(), "odemodel.pth")
print("Saved PyTorch Model State to odemodel.pth")

