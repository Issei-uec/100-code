import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

print(list[0:8])

"""
実行結果：
[{'surface': '一', 'base': '一', 'pos': '名詞', 'pos1': '数'}, {'surface': '\u3000', 'base': '\u3000', 'pos': '記号', 'pos1': '空
白'}, {'surface': '吾輩は猫である', 'base': '吾輩は猫である', 'pos': '名詞', 'pos1': '固有名詞'}, {'surface': '。', 'base': '。', 
'pos': '記号', 'pos1': '句点'}, {'surface': '名前', 'base': '名前', 'pos': '名詞', 'pos1': '一般'}, {'surface': 'は', 'base': 'は', 'pos': '助詞', 'pos1': '係助詞'}, {'surface': 'まだ', 'base': 'まだ', 'pos': '副詞', 'pos1': '助詞類接続'}, {'surface': '無い', 
'base': '無い', 'pos': '形容詞', 'pos1': '自立'}]

リーダブルコード：
変数をわかりやすくした(p10)
if文の並びに気を付けた(p86)
"""
