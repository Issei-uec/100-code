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
from transformers import AutoTokenizer
import numpy as np
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import optim
from torch import cuda
import time
from matplotlib import pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

df = pd.read_table("/home2/y2019/o1910142/train.txt").head(100)
df_v = pd.read_table("/home2/y2019/o1910142/valid.txt").head(100)
c_dic = {"b":0, "e":1, "m":2, "t":3}

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def m_dataset(data):
    title_list = []
    for title in data["TITLE"]:
        title_list.append(title)
    dataset = tokenizer(title_list, padding=True, truncation=True, return_tensors="pt")

    dataset["acu_label"] = []
    for c in data["CATEGORY"]:
        a_list = [0, 0, 0, 0]
        a_list[c_dic[c]] = 1
        dataset["acu_label"].append(a_list)
    dataset["acu_label"] = torch.tensor(dataset["acu_label"])
    
    return dataset


from torch import nn
from transformers import AutoModel

class BERTModel(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(p=0.2)
        self.line = nn.Linear(nhid, 4)
    
    def forward(self, input_id, mask):
        outputs = self.bert(input_id, attention_mask=mask)
        logits = self.line(self.drop(outputs.pooler_output))
        soft_m = torch.softmax(logits, dim=1)
        return soft_m

tr_ids = m_dataset(df)["input_ids"]
tr_mask = m_dataset(df)["attention_mask"]
tr_label = m_dataset(df)["acu_label"]

v_ids = m_dataset(df_v)["input_ids"]
v_mask = m_dataset(df_v)["attention_mask"]
v_label = m_dataset(df_v)["acu_label"]

model = BERTModel(nhid=768).to(device)
from torch.utils.data import DataLoader, TensorDataset

train_ds = TensorDataset(tr_ids, tr_mask, tr_label)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

valid_ds = TensorDataset(v_ids, v_mask, v_label)
valid_dl = DataLoader(valid_ds, batch_size=8)

loss_fn = nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)

epoch = 30
for e in range(epoch):
    model.train()
    correct = 0
    total = 0
    for ids, mas, lab in train_dl:
        ids = ids.to(device)
        mas = mas.to(device)
        lab = lab.to(device)
        output = model(ids, mas)
        loss = criterion(output, lab.float())
        optimizer.zero_grad()
        #loss = loss_fn(output, lab)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        for i, j in zip(output, lab):
            if i.argmax(0) == j.argmax(0):
                correct += 1
    print(f"accuracy_train: {correct/len(tr_label):>7f}")      
    print(f"loss_train: {loss:>7f}")

    correct = 0
    total = 0
    model.eval()
    for ids, mas, lab in valid_dl:
        ids = ids.to(device)
        mas = mas.to(device)
        lab = lab.to(device)
        optimizer.zero_grad()
        output = model(ids, mas)
        loss = criterion(output, lab.float())
        #print(loss)
        #loss = loss_fn(output, lab)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        for i, j in zip(output, lab):
            if i.argmax(0) == j.argmax(0):
                correct += 1
    print(f"accuracy_valid: {correct/len(v_label):>7f}")      
    print(f"loss_valid: {loss:>7f}")

"""

accuracy_train: 0.420000
loss_train: 0.660681
accuracy_valid: 0.420000
loss_valid: 0.660694
accuracy_train: 0.420000
loss_train: 0.785647
accuracy_valid: 0.420000
loss_valid: 0.660700
accuracy_train: 0.420000
loss_train: 0.660682
accuracy_valid: 0.420000
loss_valid: 0.660708

"""