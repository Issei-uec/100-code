import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np

dic = {}
#出現頻度を表す辞書
dic_label = {}
#ラベルを表す辞書
df = pd.read_table("/home2/y2019/o1910142/train.txt")


for title in df["TITLE"]:
    word_l = title.split()
    for word in word_l:
        if word in dic:
            dic[word] += 1
        else:
            dic[word] = 1  


dic_sort = sorted(dic.items(), key = lambda x : x[1], reverse=True)


i = 1
word_am = 0
for word in dic_sort:
    if word[1] != 1 and word[1] != 0:
        dic_label[word[0]] = i
    else:
        dic_label[word[0]] = 0
    
    i += 1
    word_am += 1

print(word_am)

def return_id(text):
    id_list = []
    word_l = text.split()
    for word in word_l:
        if word in dic_label:
            id_list.append(dic_label[word])
        else:
            id_list.append(0)
    
    return id_list


"""
実行結果
print(return_id("What we do to watch TV"))
23650
[105, 5513, 3259, 1, 1478, 162]
"""