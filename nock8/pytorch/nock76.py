import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.linear = nn.Linear(300, 4, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return torch.softmax(x, dim=1)


X = torch.load("train_x.pt")
Y = torch.load("train_y.pt")
X_v = torch.load("valid_x.pt")
Y_v = torch.load("valid_y.pt")
X_t = torch.load("test_x.pt")
Y_t = torch.load("test_y.pt")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
X = X.to(device)
Y = Y.to(device)
X_v = X_v.to(device)
Y_v = Y_v.to(device)
X_t = X_t.to(device)
Y_t = Y_t.to(device)

model = network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)

def calculate(model, X, Y):
    pred = model(X)
    total = 0
    correct = 0
    total += len(Y)
    correct += (pred.argmax(1) == Y).sum().item()
    return(correct/len(Y))

def CE_loss(X, Y, W):
    CE = 0
    X = X @ W
    for i in range(len(Y)):
        CE += -torch.log(torch.softmax(X, dim=1)[i][int(Y[i])])
    
    return(f"{CE/len(Y):.4f}")

epoch = 100
for e in range(epoch):
    pred = model(X)
    loss = loss_fn(pred, Y)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.item()

    #最終的なものだけファイルに書き出す
    #本来は、"nock76.pt"をf"nock76_e(epoch).pt"のようにする
    torch.save({"重み": model.state_dict()['linear.weight'].T, "予測": pred, "損失": loss, "内部状態": optimizer.state_dict()}, "nock76.pt")

print(torch.load("nock76.pt"))


"""
実行結果

{'重み': tensor([[-0.0897,  0.2357, -0.0297, -0.0326],
        [ 0.0500,  0.0322, -0.0628,  0.0049],
        [ 0.0490, -0.2143,  0.0333,  0.1264],
        ...,
        [ 0.2234, -0.2489,  0.0364,  0.0455],
        [ 0.2048,  0.0151, -0.0540, -0.0616],
        [-0.2662,  0.1143,  0.0807, -0.0117]], device='cuda:0'), '予測': tensor([[0.0060, 0.9864, 0.0035, 0.0040],
        [0.8085, 0.1292, 0.0293, 0.0330],
        [0.0718, 0.8968, 0.0141, 0.0174],
        ...,
        [0.1601, 0.7373, 0.0496, 0.0530],
        [0.9215, 0.0436, 0.0165, 0.0184],
        [0.1551, 0.8021, 0.0190, 0.0238]], device='cuda:0', requires_grad=True), '損失': 1.0443288087844849, '内部状態': {'state': {}, 'param_groups': [{'lr': 0.9, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [139785391978928]}]}}
"""

