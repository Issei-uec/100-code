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

saki = 0
s_list = []
morph = []
chunk = None
chunk1 = []
AI = open("ai.ja.txt.parsed", "r", encoding="utf-8")
for line in AI:
    if line[0] == "*":
        word = re.split(r',|\s', line)
        if len(morph) > 0:
            chunk = Chunk(morph, saki)
            chunk1.append(chunk)
            morph = []
        saki = int(re.search(r'(.*?)D', word[2]).group(1))
    elif line != "EOS\n":
        morph.append(Morph(line))
    else:
        chunk1.append(Chunk(morph, saki))
        if chunk1:
            for i, c in enumerate(chunk1, 0):
                if c.dst != -1:
                    chunk1[c.dst].srcs.append(i)
        s_list.append(chunk1)
        chunk1 = []               
        morph = [] 
        word = None
        chunk = None
#else:


AI.close()

for i in s_list[2]:
    print([j.surface for j in i.morphs], i.dst, i.srcs)
