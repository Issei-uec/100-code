import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
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



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

df = pd.read_table("/home2/y2019/o1910142/train.txt")
df_v = pd.read_table("/home2/y2019/o1910142/valid.txt")
c_dic = {"b":0, "e":1, "m":2, "t":3}

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def m_dataset(data):
    title_list = []
    for title in data["TITLE"]:
        title_list.append(title)
    dataset = tokenizer(title_list, padding=True, truncation=True, max_length=30, return_tensors="pt")

    a_list = []
    for c in data["CATEGORY"]:
        a_list.append(c_dic[c])
    dataset["acu_label"] = torch.tensor(a_list)
    return dataset
    

from torch import nn
from transformers import AutoModel

class BERTModel(nn.Module):
    def __init__(self, nhid):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
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
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

valid_ds = TensorDataset(v_ids, v_mask, v_label)
valid_dl = DataLoader(valid_ds, batch_size=32)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.9)

epoch = 10
for e in range(epoch):
    model.train()
    correct = 0
    total = 0
    for ids, mas, lab in train_dl:
        ids = ids.to(device)
        mas = mas.to(device)
        lab = lab.to(device)
        output = model(ids, mas)
        optimizer.zero_grad()
        loss = loss_fn(output, lab)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        correct += (output.argmax(1) == lab).sum().item()
    print(f"accuracy_train: {correct/len(tr_label):>7f}")      
    print(f"loss_train: {loss:>7f}")

    correct = 0
    total = 0
    model.eval()
    for ids, mas, lab in valid_dl:
        ids = ids.to(device)
        mas = mas.to(device)
        lab = lab.to(device)
        output = model(ids, mas)
        loss = loss_fn(output, lab)
        loss = loss.item()
        correct += (output.argmax(1) == lab).sum().item()
    print(f"accuracy_valid: {correct/len(v_label):>7f}")      
    print(f"loss_valid: {loss:>7f}")

"""
accuracy_train: 0.416698
loss_train: 1.306168
accuracy_valid: 0.437031
loss_valid: 1.289123
accuracy_train: 0.420821
loss_train: 1.368668
accuracy_valid: 0.437031
loss_valid: 1.289123
accuracy_train: 0.420821
loss_train: 1.306168
accuracy_valid: 0.437031
loss_valid: 1.289123


tensor([  101,  1054, 13876,  1011,  1999,  1996,  4439,  2835,  1024,  2859,
         1005,  1055,  9805,  9397,  3111,  2024,  2047,  3006,  2486,  2005,
         3795,  1012,  1012,  1012,   102,     0,     0,     0,     0,     0]) 
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]) 
tensor(0)

softmax
tensor([[1.0000e+00, 2.7854e-07, 5.0909e-10, 4.8373e-11],
        [1.0000e+00, 2.9476e-07, 1.4766e-09, 1.3571e-10],
        [1.0000e+00, 1.6997e-07, 4.7644e-10, 4.0044e-11],
        [1.0000e+00, 7.5900e-07, 1.8381e-09, 1.5435e-10],
        [1.0000e+00, 3.9538e-07, 4.4867e-10, 6.7366e-11],
        [1.0000e+00, 9.3740e-07, 1.1648e-09, 1.6168e-10],
"""