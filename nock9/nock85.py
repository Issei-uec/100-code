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


df = pd.read_table("/home2/y2019/o1910142/train.txt")
df_v = pd.read_table("/home2/y2019/o1910142/valid.txt")

def return_id(text, dic_label):
    id_l = []
    #idを収納
    word_l = text.split()
    #単語を収納
    for word in word_l:
        if word in dic_label:
            id_l.append(dic_label[word])
        else:
            id_l.append(0)
    
    return id_l

dic = {}
#単語と出現回数の辞書
dic_label = {}
#単語とラベルの辞書
for title in df["TITLE"]:
    word_l = title.split()
    for word in word_l:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1  

dic_sort = sorted(dic.items(), key = lambda x : x[1], reverse=True)

i = 1
#ラベル
max_len = 0
#ラベルがついた全単語数
word_am = 0
#全単語数
for word in dic_sort:
    if word[1] != 1 and word[1] != 0:
        dic_label[word[0]] = i
        max_len += 1
    else:
        dic_label[word[0]] = 0
    
    i += 1
    word_am += 1

def clean(data_flame):
    global dic_label
    vocab_size = max_len+1
    # 埋め込む次元
    emb_dim = 300
    embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    word_label_list = []
    #タイトルにおける単語のラベルのリスト
    w_len = []
    #単語ごとのラベルの長さ
    for idx in range(len(data_flame["TITLE"])):
        r = return_id(data_flame.iloc[idx]["TITLE"], dic_label)
        word_label_list.append(torch.tensor(r))
        w_len.append(len(r))
    w_len = torch.tensor(w_len)
    padded_word = pad_sequence(word_label_list, batch_first=True)
    #embed_word = embeddings(padded_word)
    #packed_emb_X = pack_padded_sequence(embed_word, w_len, enforce_sorted=False, batch_first=True)

    return(w_len, word_am, padded_word)
#(単語ごとのラベルの長さ, 全単語数, パディングした単語ラベルのリスト)

train_d = clean(df)
valid_d = clean(df_v)

word_vec_list = []

#重みづけをword2vecで
import gensim
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('/home2/y2019/o1910142/GoogleNews-vectors-negative300.bin.gz', binary=True)
for key in dic_label.keys(): 
    if key in w2v_model:
        word_vec_list.append(w2v_model[key].tolist())
    else:
        word_vec_list.append(np.random.rand(300).tolist())

word_vec_list = torch.tensor(word_vec_list)

class LSTMModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(LSTMModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight = nn.Parameter(word_vec_list)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(nhid*2, 4)                          #LSTMの双方向化
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, s_len):
        emb = self.encoder(input)
        emb_pad = pack_padded_sequence(emb, s_len, enforce_sorted=False, batch_first=True)
        output, (h_t, c_t) = self.rnn(emb_pad)
        decoded = self.decoder(torch.cat([h_t[0], h_t[-1]], dim=1)).squeeze(0)
        soft_m = torch.softmax(decoded, dim=1)
        return soft_m

model = LSTMModel(
    ntoken=train_d[1], 
    ninp=300, 
    nhid=50, 
    nlayers=1, 
).to(device)

c_dic = {"b":0, "e":1, "m":2, "t":3}
y_list = torch.tensor([c_dic[c] for c in df["CATEGORY"]])
y_list_v = torch.tensor([c_dic[c] for c in df_v["CATEGORY"]])

from torch.utils.data import DataLoader, TensorDataset

train_ds = TensorDataset(train_d[2], y_list, train_d[0])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

valid_ds = TensorDataset(valid_d[2], y_list_v, valid_d[0])
valid_dl = DataLoader(valid_ds, batch_size=64)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)

epoch = 100
for e in range(epoch):
    model.train()
    correct = 0
    for p, y, w in train_dl:
        p = p.to(device)
        y = y.to(device)
        w = w.to(device)
        pred = model(p, w)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        correct += (pred.argmax(1) == y).sum().item()
    print("epoch: ", e)
    print(f"accuracy_train: {correct/len(y_list):>7f}")      
    print(f"loss_train: {loss:>7f}")

    correct = 0
    model.eval()
    for p, y, w in valid_dl:
        p = p.to(device)
        y = y.to(device)
        w = w.to(device)
        pred = model(p, w)
        loss = loss_fn(pred, y)
        loss = loss.item()
        correct += (pred.argmax(1) == y).sum().item()
    print(f"accuracy_valid: {correct/len(y_list_v):>7f}")      
    print(f"loss_valid: {loss:>7f}")

"""
実行結果
layer=1
epoch:  99
accuracy_train: 0.971327
loss_train: 0.802629
accuracy_valid: 0.823838
loss_valid: 0.953482

layer=2
epoch:  99
accuracy_train: 0.964018
loss_train: 0.743692
accuracy_valid: 0.843328
loss_valid: 0.886714

layer=3
epoch:  99
accuracy_train: 0.958396
loss_train: 0.764868
accuracy_valid: 0.817841
loss_valid: 0.854971

layer=4
epoch:  99
accuracy_train: 0.960551
loss_train: 0.785365
accuracy_valid: 0.822339
loss_valid: 0.847357

layer=10
epoch:  99
accuracy_train: 0.958958
loss_train: 0.806218
accuracy_valid: 0.816342
loss_valid: 0.912950
"""