#41が直前までうまくいかなかったため、～49まで解答で行っている
#時間のある時に自作に変更予定
path = "ai.ja.txt.parsed"
import re
with open(path, encoding='utf-8') as f:
  _data = f.read().split('\n')

class Morph:
  def __init__(self, word):
    self.surface = word[0]
    self.base = word[7]
    self.pos = word[1]
    self.pos1 = word[2]
#クラスChunk
class Chunk:
  def __init__(self, idx, dst):
    self.idx = idx     #文節番号
    self.morphs = []   #形態素（Morphオブジェクト）のリスト
    self.dst = dst     #係り先文節インデックス番号
    self.srcs = []     #係り元文節インデックス番号のリスト

import re
#1文ごとのリスト
s_list = []
#Chunkオブジェクト
sent = []
#形態素解析結果のMorphオブジェクトリスト
temp = []
chunk = None
for line in _data[:-1]:
  #集合[]で「\t」と「,」と「　(スペース)」を区切りを指定します。
  text = re.split("[\t, ]", line)

  #係り受け解析の行の処理
  if text[0] == '*':
    idx = int(text[1])
    dst = int(re.search(r'(.*?)D', text[2]).group(1))
    #Chunkオブジェクトへ
    chunk = Chunk(idx, dst)
    sent.append(chunk)

  #EOSを目印に文ごとにリスト化
  elif text[0] == 'EOS':
    if sent:
      for i, c in enumerate(sent, 0):
        if c.dst == -1:
          continue
        else:
          sent[c.dst].srcs.append(i)
      s_list.append(sent)
    sent = []
  else:
    morph = Morph(text)
    chunk.morphs.append(morph)
    temp.append(morph)

moto = ""
saki = ""
flag_n = 0
flag_v = 0
for m in s_list[1]:
    for word in m.morphs:
        if word.pos == "名詞":
            flag_n = 1
        if word.pos != "記号":
            moto += word.surface
    for word in s_list[1][int(m.dst)].morphs:
        if word.pos == "動詞":
            flag_v = 1
        if word.pos != "記号":
            saki += word.surface
    if flag_n ==1 and flag_v == 1:
        print(moto + '\t' + saki)
    moto = saki = ""
    flag_n = flag_v = 0

"""
実行結果：
道具を  用いて
知能を  研究する    
一分野を        指す
知的行動を      代わって
人間に  代わって
コンピューターに        行わせる
研究分野とも    される

"""