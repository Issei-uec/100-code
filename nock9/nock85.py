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
    embed_word = embeddings(padded_word)
    packed_emb_X = pack_padded_sequence(embed_word, w_len, enforce_sorted=False, batch_first=True)

    return(w_len, word_am, padded_word)

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
    """エンコーダー、再帰モジュール、そしてデコーダーを含むモデル構成"""

    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(LSTMModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.weight = nn.Parameter(word_vec_list)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, batch_first=True, bidirectional=True)
        self.decoder = nn.Linear(nhid, 4)                          #LSTMの双方向化
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, s_len):
        emb = self.encoder(input)
        emb_pad = pack_padded_sequence(emb, s_len, enforce_sorted=False, batch_first=True)
        output, (h_t, c_t) = self.rnn(emb_pad)
        decoded = self.decoder(h_t[1]).squeeze(0)
        soft_m = torch.softmax(decoded, dim=1)
        return soft_m

model = LSTMModel(
    ntoken=train_d[1], 
    ninp=300, 
    nhid=50, 
    nlayers=4, 
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

epoch = 10
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
accuracy_train: 0.510120
loss_train: 1.070947
accuracy_valid: 0.591454
loss_valid: 1.055641
accuracy_train: 0.657702
loss_train: 1.125241
accuracy_valid: 0.691904
loss_valid: 0.977851
accuracy_train: 0.715049
loss_train: 0.906028
accuracy_valid: 0.727886
loss_valid: 0.959604
accuracy_train: 0.742410
loss_train: 1.017369
accuracy_valid: 0.728636
loss_valid: 0.963780
accuracy_train: 0.760588
loss_train: 1.032639
accuracy_valid: 0.728636
loss_valid: 0.957656
accuracy_train: 0.775675
loss_train: 0.915876
accuracy_valid: 0.745877
loss_valid: 0.943158
accuracy_train: 0.784108
loss_train: 0.927274
accuracy_valid: 0.753373
loss_valid: 0.942888
accuracy_train: 0.788887
loss_train: 0.969991
accuracy_valid: 0.759370
loss_valid: 0.939574
accuracy_train: 0.794790
loss_train: 0.866034
accuracy_valid: 0.755622
loss_valid: 0.933790
accuracy_train: 0.799850
loss_train: 0.930733
accuracy_valid: 0.754123
loss_valid: 0.934336


layer=1, h_t[1]
accuracy_train: 0.524457
loss_train: 1.023673
accuracy_valid: 0.601949
loss_valid: 1.081128
accuracy_train: 0.670352
loss_train: 0.889725
accuracy_valid: 0.710645
loss_valid: 0.964193
accuracy_train: 0.726387
loss_train: 0.936749
accuracy_valid: 0.686657
loss_valid: 0.990082
accuracy_train: 0.750750
loss_train: 0.954093
accuracy_valid: 0.739880
loss_valid: 0.961598
accuracy_train: 0.768834
loss_train: 0.920411
accuracy_valid: 0.736132
loss_valid: 0.969792
accuracy_train: 0.781109
loss_train: 1.004969
accuracy_valid: 0.754123
loss_valid: 0.945947
accuracy_train: 0.789262
loss_train: 1.035703
accuracy_valid: 0.759370
loss_valid: 0.943492
accuracy_train: 0.795915
loss_train: 1.093728
accuracy_valid: 0.758621
loss_valid: 0.942017
accuracy_train: 0.799569
loss_train: 1.050505
accuracy_valid: 0.757871
loss_valid: 0.935617
accuracy_train: 0.802567
loss_train: 0.953156
accuracy_valid: 0.764618
loss_valid: 0.931517

layer = 4
accuracy_train: 0.531016
loss_train: 1.140035
accuracy_valid: 0.630435
loss_valid: 1.067011
accuracy_train: 0.674569
loss_train: 0.956516
accuracy_valid: 0.577961
loss_valid: 1.145085
accuracy_train: 0.729573
loss_train: 1.002089
accuracy_valid: 0.730885
loss_valid: 0.952171
accuracy_train: 0.751124
loss_train: 0.926150
accuracy_valid: 0.735382
loss_valid: 0.948155
accuracy_train: 0.769678
loss_train: 0.947146
accuracy_valid: 0.748126
loss_valid: 0.955318
accuracy_train: 0.780360
loss_train: 0.994216
accuracy_valid: 0.744378
loss_valid: 0.941888
accuracy_train: 0.791792
loss_train: 1.003177
accuracy_valid: 0.766117
loss_valid: 0.949977
accuracy_train: 0.795821
loss_train: 0.972001
accuracy_valid: 0.764618
loss_valid: 0.942278
accuracy_train: 0.800975
loss_train: 0.952403
accuracy_valid: 0.763118
loss_valid: 0.948930
accuracy_train: 0.804442
loss_train: 1.052058
accuracy_valid: 0.766867
loss_valid: 0.947126
"""