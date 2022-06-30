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

epoch = 100
for e in range(epoch):
    pred = model(X)
    loss = loss_fn(pred, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.item()



print("train正解率：" , calculate(model, X, Y))
print("valid正解率：" , calculate(model, X_v, Y_v))


"""
実行結果

train正解率： 0.776424287856072
valid正解率： 0.7961019490254873
"""