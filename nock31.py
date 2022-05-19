import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

for i in list:
    if i["pos"] == "動詞":
        print(i["surface"])

"""
実行結果：
見える
申し
願う
下さっ
下れ
這入り
かから
来
落ちつい
足り
なっ
着か
いる
思う
着せる

リーダブルコード：

"""