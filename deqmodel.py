import torch
import torch.nn as nn
import torch.autograd as autograd

device = "cuda" if torch.cuda.is_available() else "cpu"

# DEQFixedPoint and anderson code taken from the DEQs repo
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)['result']
        z = self.f(z, x)

        if self.training:
            # set up Jacobian vector product (without additional forward calls)
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(z0, x)

            def backward_hook(grad):
                g = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                   grad, **self.kwargs)['result']
                return g

            z.register_hook(backward_hook)
        return z


def anderson(f, x0, m=6, lam=1e-4, threshold=50, eps=1e-3, stop_mode='rel', beta=1.0, **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, L = x0.shape
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)

        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest, lowest_gx = X[:, k % m].view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {"result": lowest_xest,
           "lowest": lowest_dict[stop_mode],
           "nstep": lowest_step_dict[stop_mode],
           "prot_break": False,
           "abs_trace": trace_dict['abs'],
           "rel_trace": trace_dict['rel'],
           "eps": eps,
           "threshold": threshold}
    X = F = None
    return out


class Func(nn.Module):
    def __init__(self, data_size, hidden_size):
        super(Func, self).__init__()
        self.h_layer = nn.Linear(data_size, hidden_size)
        self.x_layer = nn.Linear(data_size, hidden_size)
        self.tanh = nn.Tanh()
        self.output = nn.Sequential(nn.Linear(hidden_size, data_size), nn.Tanh())

    def forward(self, x, z):
        hidden = self.tanh(self.h_layer(z) + self.x_layer(x))
        output = self.output(hidden)
        return output


class NeuralNetwork(nn.Module):
    def __init__(self, data_size=14, seq_len=31, hidden_size=50, output_size=1, batch_size=64):
        super(NeuralNetwork, self).__init__()

        func = Func(data_size, hidden_size)

        self.mydeq = DEQFixedPoint(func, solver=anderson)

        self.fixoutput = nn.Sequential(
            nn.Flatten(),
            nn.Linear(data_size*seq_len, output_size))

    def forward(self, x):
        x = self.mydeq(x)
        return self.fixoutput(x)


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