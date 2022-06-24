import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import math


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.linear = nn.Linear(300, 4, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return torch.softmax(x, dim=1)


X = torch.load("train_x.pt")[0:100]
Y = torch.load("train_y.pt")[0:100]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
X = X.to(device)
Y = Y.to(device)

print(X.shape)
print(Y.shape)

model = network().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)

epoch = 100
for e in range(epoch):
    pred = model(X)
    loss = loss_fn(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.item()
    print(f"loss: {loss:>7f}")

