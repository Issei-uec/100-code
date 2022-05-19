import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

N_in = []
no_in = []
for i in list:
    if len(N_in) != 0:
        if i["pos"] == "名詞":        
            if len(no_in) != 0:
                print(N_in[0] + i["surface"])
                N_in = []
                no_in = []
        elif i["surface"] == "の":
            no_in = [i["surface"]]
        else: 
            N_in = []
            no_in = []
    elif i["pos"] == "名詞":
        N_in = [i["surface"]]

    

