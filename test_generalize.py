from generate_data import generate_longer_data, generate_biased_data
import torch
from torch import nn
import basemodel
import deqmodel
from torch.utils.data import DataLoader
import numpy as np

test_data = generate_biased_data()
print("Generated test data!")
filenames = ["basemodel_exp.pth", "mlmodel_exp.pth", "deqmodel_exp.pth"]

for name in filenames:
    print(f"Loading from {name}...")
    state_dict = torch.load(name)
    if "basemodel" in name:
        model = basemodel.NeuralNetwork()
        modeltype = basemodel
    elif "mlmodel" in name:
        model = basemodel.NeuralNetwork(num_layers=3)
        modeltype = basemodel
    else:
        model = deqmodel.NeuralNetwork()
        modeltype = deqmodel

    model.load_state_dict(state_dict)

    pos_weight = torch.from_numpy(np.array([0.66])).to(modeltype.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(modeltype.device)

    test_dataloader = DataLoader(test_data, batch_size=64)
    modeltype.test(test_dataloader, model, loss_fn)



