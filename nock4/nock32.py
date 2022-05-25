import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

for word in list:
    if word["pos"] == "動詞":
        print(["base"])
word
"""
実行結果：
鳴らす
する
しまう
怒鳴る
怒鳴る
怒鳴る
いる
云う
する
出す
聞く

リーダブルコード：
変数をわかりやすくした(p10)
if文の並びに気を付けた(p86)
"""