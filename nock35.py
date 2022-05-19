import re

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
    if word["surface"] != "。" and word["surface"] != "、" and word["surface"] != "「" and word["surface"] != "」":
        if not word["surface"] in dic_word:
            dic_word[word["surface"]] = 1
        else:
            dic_word[word["surface"]] += 1

sort_dic_word = sorted(dic_word.items(), key=lambda x:x[1], reverse=True)
print(sort_dic_word)