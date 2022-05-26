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
for m in s_list[1]:
    for word in m.morphs:
        if word.pos != "記号":
            moto += word.surface
    for word in s_list[1][int(m.dst)].morphs:
        if word.pos != "記号":
            saki += word.surface
    print(moto + '\t' + saki)
    moto = saki = ""


"""
実行結果：
人工知能        語
じんこうちのう  語
AI      エーアイとは
エーアイとは    語
計算    という
という  道具を
概念と  道具を
コンピュータ    という
という  道具を
道具を  用いて

"""