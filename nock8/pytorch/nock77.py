import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import math
import time

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
print(elapsed_time)


"""
実行結果
epoch5で行った
train正解率： 1024 0.7729572713643178
0.1520709991455078
train正解率： 128 0.7836394302848576
0.3412449359893799
train正解率： 32 0.7889805097451275
1.0873055458068848
train正解率： 16 0.7918853073463268
2.102994918823242
train正解率： 8 0.8899925037481259
4.030369758605957
train正解率： 4 0.9032983508245878
8.12349247932434
train正解率： 2 0.9129497751124438
15.745636463165283
train正解率： 1 0.9196026986506747
31.21795892715454
"""