import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('newsCorpora.csv', sep = "\t", names=["ID","TITLE","URL","PUBLISHER","CATEGORY","STORY","HOSTNAME","TIMESTAMP"])
df = df[(df["PUBLISHER"] == "Reuters") | (df["PUBLISHER"] == "Huffington Post") | (df["PUBLISHER"] == "Businessweek") | (df["PUBLISHER"] == "Contactmusic.com") | (df["PUBLISHER"] == "Daily Mail")]
df1 = df[["TITLE", "CATEGORY"]]
df2 = df1.replace('"', '')

print(df1["TITLE"].head(5))

train, valid = train_test_split(df2, test_size=0.2)
valid, test = train_test_split(valid, test_size=0.5)

#label 0:b 1:e 2:m 3:t

print("train")
print("b" + "\t", len(train[train["CATEGORY"] == "b"]))
print("e" + "\t", len(train[train["CATEGORY"] == "e"]))
print("m" + "\t", len(train[train["CATEGORY"] == "m"]))
print("t" + "\t", len(train[train["CATEGORY"] == "t"]))

print("\n" + "valid")
print("b" + "\t", len(valid[valid["CATEGORY"] == "b"]))
print("e" + "\t", len(valid[valid["CATEGORY"] == "e"]))
print("m" + "\t", len(valid[valid["CATEGORY"] == "m"]))
print("t" + "\t", len(valid[valid["CATEGORY"] == "t"]))

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

import torch
def word_vec(text):
  vec_list_all = []
  for title in text:
    vec_list = []
    #多分ミス
    title = ' '.join(char for char in title if char.isalnum())
    word_list = title.split()
    for word in word_list:
      if word != "a":
        vec_list.append(model[word])
    vec_list_all.append(sum(vec_list)/len(vec_list))

  return torch.tensor(vec_list_all)

def label(text):
  label_dic = {"b":0, "e":1, "m":2, "t":3}
  label_list = []
  for category in text:
    label_list.append(label_dic[category])
  return torch.tensor(label_list)

torch.save(word_vec(train["TITLE"]), "train_x.pt")
torch.save(word_vec(valid["TITLE"]), "valid_x.pt")
torch.save(word_vec(test["TITLE"]), "test_x.pt")

torch.save(label(train["CATEGORY"]), "train_y.pt")
torch.save(label(valid["CATEGORY"]), "valid_y.pt")
torch.save(label(test["CATEGORY"]), "test_y.pt")
