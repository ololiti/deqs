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
        self.rnn = nn.RNN(data_size, hidden_size, batch_first=True, num_layers=3)
        self.fixoutput = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size*seq_len, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        # self.final = nn.Linear(hidden_size, output_size)
        # nn.init.xavier_uniform_(self.final.weight)

    def forward(self, x):
        # h0 = self.initHidden()
        output, hidden = self.rnn(x)
        output = self.fixoutput(output)
        # output = self.final(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(size=(1, self.batch_size, self.hidden_size)).float()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()

        optimizer.zero_grad()
        # Compute prediction error
        pred, hidden = model(X)
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
        first = True
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device).float()
            pred, hidden = model(X)
            # print(f"y values: {y[0:10]}")
            # print(f"predicted y values: {pred[0:10]}")
            if first:
                print(f"first prediction: {pred[0:5]}, y val: {y[0:5]}")
                first = False
            test_loss += loss_fn(pred, y).item()
            correct += (abs(pred - y) < 0.2).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct