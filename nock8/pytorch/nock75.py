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

accu_l =  []
loss_l = []
accu_lv =  []
loss_lv = []
epoch = 100
epoch_l = range(1, epoch+1)
for e in range(epoch):
    pred = model(X)
    loss = loss_fn(pred, Y)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.item()

    accu_l.append(calculate(model, X, Y))
    #loss_l.append(CE_loss(X, Y, model.state_dict()['linear.weight'].T))
    loss_l.append(loss)
    accu_lv.append(calculate(model, X_v, Y_v))
    pred = model(X_v)
    loss = loss_fn(pred, Y_v)   
    loss_lv.append(loss.item())

plt.figure()
plt.plot(epoch_l, accu_l)
plt.plot(epoch_l, accu_lv)
plt.savefig("nock75_accu.png")


plt.figure()
plt.plot(epoch_l, loss_l)
plt.plot(epoch_l, loss_lv)
plt.savefig("nock75_loss.png")



