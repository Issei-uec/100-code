import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

print(list[4:8])

"""
実行結果：
[{'surface': '名前', 'base': '名前', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助
詞'}, {'surface': 'まだ', 'base': 'まだ', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': '無い', 'base': '無い', 'pos': '形容詞
', 'pos1': '自立'}]

リーダブルコード：

"""
