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

for word in list:
    if word["pos"] == "名詞":
        noun_list.append(word["surface"])
        max_len = len(noun_list)
    
    else:
        if max_len > 3:
            print(noun_list)
        noun_list = []
        max_len = 0


"""
実行結果：
['一', '字', '一', '句']
['吾輩', '自ら', '余', '瀾']
['両人', '共', '応対', '振り']
['落', '雲', '館', '事件']
['今日', '何', '人', 'あばた']
['幅', '三', '尺', '八', '寸', '高さ', 'これ']
['改良', '首', 'きり', '器械']

リーダブルコード：
変数をわかりやすくした(p10)
if文の並びに気を付けた(p86)
"""
