import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, data_size=14, seq_len=31, hidden_size=50, output_size=1, batch_size=64):
        super(NeuralNetwork, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(data_size*seq_len, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(self.layers(self.flatten(x)))


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
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
            X, y = X.to(device).float(), y.to(device).float()
            pred = model(X)
            #print(f"first prediction: {pred[0]}, y val: {y[0]}")
            test_loss += loss_fn(pred, y).item()
            correct += (abs(torch.sigmoid(pred) - y) < 0.5).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct




