import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import time

class network(nn.Module):
    """
    def __init__(self):
        super(network, self).__init__()
        self.linear = nn.Linear(300, 4, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return torch.softmax(x, dim=1)

    """
    
    def __init__(self):
        super(network, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  
        )
    
    """    
    def __init__(self):
        super(network, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),    
        )
    """
    #"""
    def forward(self, x):
        x = self.linear_relu_stack(x)
        return torch.softmax(x, dim=1)

    #"""
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
batch_size = 64
X_list = torch.split(X, batch_size)
Y_list = torch.split(Y, batch_size)
j = 0

t1 = time.time() 
for e in range(epoch):
    for i in range(len(X_list)):
        pred = model(X_list[i])
        loss = loss_fn(pred, Y_list[i])  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        

t2 = time.time() 
elapsed_time = t2-t1
    #torch.save({"重み": model.state_dict()['linear.weight'].T, "予測": pred, "損失": loss, "内部状態": optimizer.state_dict()}, "nock76.pt")
print("train正解率：" , batch_size, calculate(model, X, Y))
print("valid正解率：" , batch_size, calculate(model, X_v, Y_v))
print("test正解率：" , batch_size, calculate(model, X_t, Y_t))
print(elapsed_time)


"""
実行結果
多層化前
Using cuda device
train正解率： 64 0.907608695652174
valid正解率： 64 0.904047976011994
test正解率： 64 0.8973013493253373
9.937106847763062

多層化後第一
Using cuda device
train正解率： 64 0.9569902548725637
valid正解率： 64 0.9100449775112444
test正解率： 64 0.9047976011994003
23.06361436843872

多層化後第二
Using cuda device
train正解率： 64 0.9337518740629686
valid正解率： 64 0.9175412293853074
test正解率： 64 0.8988005997001499
32.44616389274597
"""
