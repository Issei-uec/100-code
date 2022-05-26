import re
class Morph:
  def __init__(self, sentence):
    word = re.split(r',|\s', sentence)
    self.surface = word[0]
    self.base = word[7]
    self.pos = word[1]
    self.pos1 = word[2]

class Chunk():
  def __init__(self, morphs, dst):
    self.morphs = morphs
    self.dst = dst
    self.srcs = []

s_list = []
saki = 0
morph = []
chunk = []
chunk1 = []
AI = open("ai.ja.txt.parsed", "r", encoding="utf-8")
for line in AI:
    if line[0] == "*":
        word = re.split(r',|\s', line)
        if len(morph) > 0:
            chunk.append(Chunk(morph, saki))
            morph = []

        saki = int(re.search(r'(.*?)D', word[2]).group(1))
    elif line != "EOS\n":
        morph.append(Morph(line))
    else:
        chunk.append(Chunk(morph, saki))
        if chunk:
            for i, c in enumerate(chunk, 0):
                if c.dst != -1:
                    chunk[c.dst].srcs.append(i)
                s_list.append(chunk)
        chunk = []               
        morph = [] 
        word = None
#else:


AI.close()
"""
for i in s_list[3]:
    print([j.surface for j in i.morphs], i.dst, i.srcs)
"""
for i in range(3):
    print(s_list[i])