import torch
import torch.nn as nn
import torch.autograd as autograd

device = "cuda" if torch.cuda.is_available() else "cpu"

# DEQFixedPoint and anderson code taken from the DEQs repo
class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, hidden_size, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.hidden_size = hidden_size
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z_shape = (x.size(dim=0), x.size(dim=1), self.hidden_size)
            z = self.solver(lambda z: self.f(z, x), x0=x, **self.kwargs)['result']
        z = self.f(z, x)

        if self.training:
            # set up Jacobian vector product (without additional forward calls)
            z0 = z.clone().detach().requires_grad_()
            z0 = z0.to(device)
            f0 = self.f(z0, x)
            f0 = f0.to(device)

            def backward_hook(grad):
                g = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                   grad, **self.kwargs)['result']
                g = g.to(device)
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

def broyden(f, x0, threshold, eps=1e-3, stop_mode="rel", ls=False, name="unknown"):
    bsz, total_hsize, seq_len = x0.size()
    g = lambda y: f(y) - y
    dev = x0.device
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    
    x_est = x0           # (bsz, 2d, L')
    gx = g(x_est)        # (bsz, 2d, L')
    nstep = 0
    tnstep = 0
    
    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, threshold).to(dev)     # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, threshold, total_hsize, seq_len).to(dev)
    update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)      # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False
    
    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len
    new_objective = 1e8

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode: 
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        new_objective = diff_dict[stop_mode]
        if new_objective < eps: break
        if new_objective < 3*eps and nstep > 30 and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:,:,:,:nstep-1], VTs[:,:nstep-1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:,None,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,nstep-1] = vT
        Us[:,:,:,nstep-1] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)

    # Fill everything up to the threshold length
    for _ in range(threshold+1-len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return {"result": lowest_xest,
            "lowest": lowest_dict[stop_mode],
            "nstep": lowest_step_dict[stop_mode],
            "prot_break": prot_break,
            "abs_trace": trace_dict['abs'],
            "rel_trace": trace_dict['rel'],
            "eps": eps,
            "threshold": threshold}


class Func(nn.Module):
    def __init__(self, data_size, hidden_size):
        super(Func, self).__init__()
        self.rnnz = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.tanh = nn.Tanh()

    def forward(self, z, x):
        z, x = z.to(device), x.to(device)
        outputz, hiddenz = self.rnnz(z)
        final = self.tanh(outputz + x)
        return final


class NeuralNetwork(nn.Module):
    def __init__(self, data_size=14, seq_len=31, hidden_size=50, output_size=1, batch_size=64):
        super(NeuralNetwork, self).__init__()

        func = Func(data_size, hidden_size)
        
        self.rnnx = nn.GRU(data_size, hidden_size, batch_first=True)


        self.mydeq = DEQFixedPoint(func, solver=broyden, hidden_size=hidden_size)

        self.fixoutput = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * seq_len, output_size)
        )

    def forward(self, x):
        x = x.to(device)
        z0, hidden = self.rnnx(x)
        output = self.mydeq(z0)
        return self.fixoutput(output)


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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


