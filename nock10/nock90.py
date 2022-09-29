from curses import use_env
from decimal import DecimalException
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import optim
from torch import cuda
from transformers import BertJapaneseTokenizer

fj_train = open("/home2/y2019/o1910142/kftt-data-1.0/data/tok/kyoto-train.cln.ja", "r")
fe_train = open("/home2/y2019/o1910142/kftt-data-1.0/data/tok/kyoto-train.cln.en", "r")
fj_test = open("/home2/y2019/o1910142/kftt-data-1.0/data/tok/kyoto-test.ja", "r")
fe_test = open("/home2/y2019/o1910142/kftt-data-1.0/data/tok/kyoto-test.en", "r")

def split(text_j, text_e):
  split_list_j = []
  split_list_e = []
  for j, e in zip(text_j, text_e):
    list_j = j.split()
    list_e = e.split()
    if  len(list_j) < 30 and len(list_e) < 30:
      split_list_j.append(list_j)
      split_list_e.append(list_e)    
  return split_list_j, split_list_e

tr_jlist, tr_elist = split(fj_train, fe_train)
te_jlist, te_elist = split(fj_test, fe_test)

fj_train.close()
fe_train.close()
fj_test.close()
fe_test.close()

def insert_s(list):
  for i in list:
    i.insert(0, "<bos>")
    i.append("<eos>")

insert_s(tr_jlist)
insert_s(te_jlist)
insert_s(tr_elist)
insert_s(te_elist)

def make_dic(list, dic, dex):
  dic["<unk>"] = 0
  dic["<pad>"] = 1
  dic["<bos>"] = 2
  dic["<eos>"] = 3

  for i in list:
    for word in i:
      if word not in dic:
        dic[word] = dex
        dex += 1
  return dic, dex

dic_j = {}
dic_e = {}
dex_j = 4
dex_e = 4
dic_j, dex_j = make_dic(te_jlist, dic_j, dex_j)
dic_e, dex_e = make_dic(te_elist, dic_e, dex_e)
dic_j, dex_j = make_dic(tr_jlist, dic_j, dex_j)
dic_e, dex_e = make_dic(tr_elist, dic_e, dex_e)

def c_dex(list, dic):
  dex_list = []
  for l in list:
    dex = []
    for word in l:
      dex.append(dic[word])
    dex_list.append(torch.tensor(dex))
  
  return dex_list



tr_dexj = c_dex(tr_jlist, dic_j)
te_dexj = c_dex(te_jlist, dic_j)
tr_dexe = c_dex(tr_elist, dic_e)
te_dexe = c_dex(te_elist, dic_e)



tr_dexj = pad_sequence(tr_dexj, batch_first=True, padding_value=1)
te_dexj = pad_sequence(te_dexj, batch_first=True, padding_value=1)
tr_dexe = pad_sequence(tr_dexe, batch_first=True, padding_value=1)
te_dexe = pad_sequence(te_dexe, batch_first=True, padding_value=1)




from torch.utils.data import DataLoader, TensorDataset

train_ds = TensorDataset(tr_dexj, tr_dexe)
train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)

valid_ds = TensorDataset(te_dexj, te_dexe)
val_dataloader = DataLoader(valid_ds, batch_size=64)

