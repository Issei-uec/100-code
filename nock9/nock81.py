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


class LSTMModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(LSTMModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, 4)
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, s_len):
        emb = self.encoder(input)
        emb_pad = pack_padded_sequence(emb, s_len, enforce_sorted=False, batch_first=True)
        output, (h_t, c_t) = self.rnn(emb_pad)
        decoded = self.decoder(h_t).squeeze(0)
        soft_m = torch.softmax(decoded, dim=1)
        return soft_m

model = LSTMModel(
    ntoken=word_am, 
    ninp=300, 
    nhid=50, 
    nlayers=1, 
)


print(model(padded_word, w_len))

"""
実行結果
tensor([[0.2140, 0.3143, 0.2061, 0.2655],
        [0.2286, 0.2687, 0.2555, 0.2472],
        [0.2010, 0.3199, 0.1861, 0.2929],
        ...,
        [0.2086, 0.2831, 0.2378, 0.2705],
        [0.2221, 0.2442, 0.2247, 0.3090],
        [0.2273, 0.2987, 0.2126, 0.2615]], grad_fn=<SoftmaxBackward>)
"""