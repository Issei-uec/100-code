import re
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

word_list = []
dic_word = {}
for word in list:
    if word["surface"] != "。" and word["surface"] != "、" and word["surface"] != "」" and word["surface"] != "「":
        if not word["surface"] in dic_word:
            dic_word[word["surface"]] = 1
        else:word_list2 = []

dic_word2 = {}
for word in list:
    if word["surface"] != "、" and word["surface"] != "」" and word["surface"] != "「":
        word_list2.append(word["surface"])

for word in list:
    if word["surface"] != "。" and word["surface"] != "、" and word["surface"] != "」" and word["surface"] != "「":
        if not word["surface"] in dic_word2:
            dic_word2[word["surface"]] = 0 

flag = 0
for i in range(len(word_list2)):
    if word_list2[i] == "猫":
        flag = 1
    elif word_list2[i] == "。":
        flag = 0
    elif flag == 1:
        dic_word2[word_list2[i]] += 1

for i in range(len(word_list2)):
    if word_list2[-i-1] == "猫":
        flag = 1
    elif word_list2[-i-1] == "。":
        flag = 0
    elif flag == 1:
        dic_word2[word_list2[-i-1]] += 1

sort_dic_word2 = sorted(dic_word2.items(), key=lambda x:x[1], reverse=True)

print(sort_dic_word2)

x = []
y = []
for i in range(10):
    x.append(sort_dic_word2[i][0])
    y.append(sort_dic_word2[i][1])

beside = np.array(x)
height = np.array(y)
plt.bar(beside, height)

"""
リーダブルコード：
変数をわかりやすくした(p10)
if文の並びに気を付けた(p86)
"""