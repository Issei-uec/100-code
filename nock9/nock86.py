import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import os
from io import open
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

dic = {}
dic_label = {}
df = pd.read_table("/home2/y2019/o1910142/train.txt")
df_v = pd.read_table("/home2/y2019/o1910142/valid.txt")

word_main_l = []
for title in df["TITLE"]:
    word_l = title.split()
    word_main_l.append(word_l)
    for word in word_l:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1  

dic_sort = sorted(dic.items(), key = lambda x : x[1], reverse=True)


i = 1
#ラベル
max_len = 0
#0でないラベルの付いた全単語
word_am = 0
#辞書中の全単語
for word in dic_sort:
    if word[1] != 1 and word[1] != 0:
        dic_label[word[0]] = i
        max_len += 1
    else:
        dic_label[word[0]] = 0
    
    i += 1
    word_am += 1

print(max_len)

def return_id(text):
    id_list = []
    word_l = text.split()
    for word in word_l:
        if word in dic_label:
            id_list.append(dic_label[word])
        else:
            id_list.append(0)
    
    return id_list

vocab_size = max_len+1
# 埋め込む次元
emb_dim = 300
embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
word = []
w_len = []
for idx in range(len(df["TITLE"])):
    r = return_id(df.iloc[idx]["TITLE"])
    word.append(torch.tensor(r))
    w_len.append(len(r))
w_len = torch.tensor(w_len)
padded_word = pad_sequence(word, batch_first=True)
#embed_word = embeddings(padded_word)
#packed_emb_X = pack_padded_sequence(embed_word, w_len, enforce_sorted=False, batch_first=True)


class CNNModel(nn.Module):

    def __init__(self, ntoken, fil, stride, ninp, nhid, nlayers):
        super(CNNModel, self).__init__()
        self.emb = nn.Embedding(ntoken, ninp)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(1, nhid, (fil, ninp), stride)
        self.line = nn.Linear(nhid, 4)
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, s_len):
        emb = self.emb(input).unsqueeze(1)   
        output = self.conv(emb)
        relu = self.relu(output) 
        pool = F.max_pool2d(relu, kernel_size=(relu.size()[2], 1))
        pool = pool.squeeze(3)
        decoded = self.line(pool.squeeze(2))
        soft_m = torch.softmax(decoded, dim=1)
        return soft_m

model = CNNModel(
    ntoken=word_am, 
    ninp=300, 
    nhid=50, 
    nlayers=1, 
    fil=3,
    stride=1
)


print(model(padded_word, w_len))

"""
実行結果
torch.Size([10672, 50, 1, 1])
tensor([[0.2774, 0.2558, 0.1984, 0.2684],
        [0.2706, 0.2809, 0.1401, 0.3084],
        [0.2810, 0.3783, 0.1236, 0.2171],
        ...,
        [0.2581, 0.3002, 0.1764, 0.2653],
        [0.3379, 0.2806, 0.1494, 0.2321],
        [0.2656, 0.2429, 0.1951, 0.2964]], grad_fn=<SoftmaxBackward>)
"""