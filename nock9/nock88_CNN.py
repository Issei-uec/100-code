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




c_dic = {"b":0, "e":1, "m":2, "t":3}
y_list = torch.tensor([c_dic[c] for c in df["CATEGORY"]])
y_list_v = torch.tensor([c_dic[c] for c in df_v["CATEGORY"]])

from torch.utils.data import DataLoader, TensorDataset

loss_fn = nn.CrossEntropyLoss()

epoch = 30
def objective(trial):
    nlayers = trial.suggest_int("nlayers", 1, 5)
    nhid = trial.suggest_int("nhid", 8, 256, 8)
    batch = trial.suggest_int("batch", 8, 128, 8)
    learning_rate = trial.suggest_discrete_uniform("learning_rate",1e-3, 1e0, q=1e-3)

    train_ds = TensorDataset(train_d[2], y_list, train_d[0])
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)

    valid_ds = TensorDataset(valid_d[2], y_list_v, valid_d[0])
    valid_dl = DataLoader(valid_ds, batch_size=batch)

    model = CNNModel(ntoken=train_d[1], ninp=300, nhid=nhid, nlayers=nlayers, fil=3, stride=1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
    return(correct/len(y_list_v))

import optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, timeout=5400)
trial = study.best_trial
print(trial)

"""
実行結果
FrozenTrial(number=106, values=[0.8328335832083958], datetime_start=datetime.datetime(2022, 7, 21, 22, 34, 36, 803764), datetime_complete=datetime.datetime(2022, 7, 21, 22, 35, 14, 123779), params={'nlayers': 1, 'nhid': 256, 'batch': 88, 'learning_rate': 0.41100000000000003}, distributions={'nlayers': IntUniformDistribution(high=5, low=1, step=1), 'nhid': IntUniformDistribution(high=256, low=8, step=8), 'batch': IntUniformDistribution(high=128, low=8, step=8), 'learning_rate': DiscreteUniformDistribution(high=1.0, low=0.001, q=0.001)}, user_attrs={}, system_attrs={}, intermediate_values={}, trial_id=106, state=TrialState.COMPLETE, value=None)
"""