import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import math

W = torch.rand(300, 4, requires_grad=True)

x = torch.load("train_x.pt")[0].unsqueeze(0) @ W
CE = torch.softmax(x, dim=1)[0][int(torch.load("train_y.pt")[0])]
print(f"事例のクロスエントロピー損失：{-torch.log(CE):.4f}")
E = -torch.log(CE)
E.backward()
print(W.grad)

W.grad.zero_()

CE = 0
X = torch.load("train_x.pt")[0:4] @ W
for i in range(4):
    CE += -torch.log(torch.softmax(X, dim=1)[i][int(torch.load("train_y.pt")[i])])

print(f"事例集合のクロスエントロピー損失：{CE/4:.4f}")
E = CE/4
E.backward()
print(W.grad)

"""
事例のクロスエントロピー損失:1.7621
事例集合のクロスエントロピー損失:1.7016

"""