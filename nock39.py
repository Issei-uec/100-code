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
        else:
            dic_word[word["surface"]] += 1

sort_dic_word = sorted(dic_word.items(), key=lambda x:x[1], reverse=True)

frequency = []
rank = []
now_rank = 1

for word in sort_dic_word:
    frequency.append(word[1])
    now_rank += 1
    rank.append(now_rank)

x = rank
y = frequency
plt.plot(x, y)

ax = plt.gca()
ax.set_yscale('log')  
ax.set_xscale('log')

plt.xlabel('出現頻度順位')
plt.ylabel('出現頻度')

plt.show()