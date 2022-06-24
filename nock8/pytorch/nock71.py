import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np

W = torch.rand(300, 4)
x = torch.load("train_x.pt")[0].unsqueeze(0) @ W
print(torch.softmax(x, dim=1))
X = torch.load("train_x.pt")[0:4] @ W
print(torch.softmax(X, dim=1))

"""
実行結果
tensor([0.1288, 0.1782, 0.3524, 0.3406])
tensor([[0.1288, 0.1782, 0.3524, 0.3406],
        [0.1227, 0.1759, 0.3881, 0.3134],
        [0.1349, 0.1938, 0.3457, 0.3256],
        [0.1088, 0.1828, 0.3716, 0.3368]])

"""