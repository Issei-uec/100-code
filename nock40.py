import re
class Morph:
  def __init__(self, sentence):
    word = re.split(r',|\s', sentence)
    self.surface = word[0]
    self.base = word[7]
    self.pos = word[1]
    self.pos1 = word[2]
  
sentence = []
list = []
AI = open("ai.ja.txt.parsed", "r", encoding="utf-8")
for line in AI:
  if line[0] != "*":
    if line != "EOS\n":
      list.append(Morph(line))
    else:
      sentence.append(list)
      list = []


AI.close()

for morph in sentence[2]:
    print(vars(morph))

"""
実行結果：
{'surface': '人工', 'base': '人工', 'pos': '名詞', 'pos1': '一般'}
{'surface': '知能', 'base': '知能', 'pos': '名詞', 'pos1': '一般'}
{'surface': '（', 'base': '（', 'pos': '記号', 'pos1': '括弧開'}
{'surface': 'じん', 'base': 'じん', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'こうち', 'base': 'こうち', 'pos': '名詞', 'pos1': '一般'}
{'surface': 'のう', 'base': 'のう', 'pos': '助詞', 'pos1': '終助詞'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': '、', 'base': '、', 'pos': '記号', 'pos1': '読点'}
{'surface': 'AI', 'base': '*', 'pos': '名詞', 'pos1': '一般'}
"""