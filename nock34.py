import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

noun_list = []
max_list = []
max_len = 0

for i in list:
    if i["pos"] == "名詞":
        noun_list.append(i["surface"])
        max_len = len(noun_list)
    
    else:
        if max_len > 3:
            print(noun_list)
        noun_list = []
        max_len = 0



