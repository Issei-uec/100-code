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
  else:
    continue

AI.close()

for morph in sentence[2]:
    print(vars(morph))