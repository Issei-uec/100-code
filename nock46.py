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


particle = []
flame_info = []
sort_flame_info = []
p_particle = ""
v_flame = ""
f = open('nock46.txt', 'w', encoding='utf-8')
for sentence in s_list:
    for chunk in sentence:
        for word in chunk.morphs:
            if word.pos == "動詞":
                for index in chunk.srcs:
                    for word_1 in sentence[index].morphs:
                        if word_1.pos == "助詞":
                            particle.append(word_1.base)                         
                            flame_info.append(''.join(word_2.surface for word_2 in sentence[index].morphs if word_2.pos != '記号'))
                if len(particle) > 0:
                  particle = sorted(list(set(particle)))
                  for p in particle:
                      p_particle += p + " "
                  for i in particle:
                    for j in flame_info:
                      if i in j and j not in sort_flame_info:
                        sort_flame_info.append(j)
                  f.write(word.base + '\t' + p_particle.rstrip() + '\t' + ' '.join(sort_flame_info)  + '\n')
                  particle = []
                  p_particle = ""
                  flame_info = []
                  sort_flame_info = []
                  v_flame = ""
f.close()

"""
実行結果：
用いる	を	道具を
する	て を	用いて 知能を
指す	を	一分野を
代わる	に を	人間に 知的行動を
行う	て に	代わって コンピューターに
せる	て に	代わって コンピューターに
する	と も	研究分野とも

"""