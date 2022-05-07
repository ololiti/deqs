import torch
from torch import nn
from torchdyn.core import NeuralODE
from torchdyn.nn import Augmenter

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        # func = nn.Sequential(nn.Tanh(),
        #                      nn.Linear(512, 512),
        #                      nn.Tanh())
        #
        # neuralDE = NeuralODE(func,
        #                      solver='rk4',
        #                      sensitivity='autograd',
        #                      return_t_eval=False)

        func = nn.Sequential(
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh()
        )
        neuralDE = NeuralODE(func, solver='rk4', sensitivity='autograd', return_t_eval=False)

        self.linearode = nn.Sequential(
            nn.Linear(28 * 28, 512), neuralDE, nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linearode(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)[-1]
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)[-1]
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct