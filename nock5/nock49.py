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


def make_word(morph):
    word_m = ""
    for word in morph.morphs:
        if word.pos != "記号":
            word_m += word.surface
    return word_m

def make_word1(morph, X):
    word_m = ""
    i = 1
    flag = 0
    for word in morph.morphs:
        if word.pos != "記号":
            if word.pos == "名詞" and i == 1:
                word_m += X
                i = 0
                flag = 1
            else:
                word_m += word.surface
    if flag == 1:           
      return word_m

def make_pass(morph, ans, list_num, j):
    for word in morph.morphs:
        if s_list[list_num][int(morph.dst)].dst == -1:
            return str(ans) + " => " + str(make_word(s_list[list_num][int(morph.dst)]))
        else:
            if s_list[list_num][int(morph.dst)].dst == int(j):
                return str(ans) + " => " + str(make_word1(s_list[list_num][int(morph.dst)], "Y"))
                
            else:
                return str(ans) + " => " + str(make_pass(s_list[list_num][int(morph.dst)], make_word(s_list[list_num][int(morph.dst)]), list_num, j)) 

#間に合わなかった
#方向性としては、iの経路上のindexをリストに入れ、jがその中にある時とない時で場合分け(問題文の場合分け)
#jが中にある場合を行おうと思ったが、上手くいかなかった
#関数を用いるとややこしい

def pass_list(index, list_num):
    for word in s_list[list_num][index].morphs:
        if s_list[list_num][index].dst == -1:
            return list_noun
        else:
            list_noun.append(s_list[list_num][index].dst)
            return pass_list(s_list[list_num][index].dst, list_num)
  
import itertools

j_list = []
list_noun = []
noun_index_list = []
Pass_list = []
for index, chunk in enumerate(s_list[1]):
    for word in chunk.morphs:
        if word.pos == "名詞":
            noun_index_list.append(index)
            
c = itertools.combinations(noun_index_list, 2)
for i, j in c:
    if i < j:
        j_list.append(j)
        list_noun.append(i)
        if j in (pass_list(i, 1)):
            Pass_list.append(make_pass(s_list[1][i], make_word1(s_list[1][i], "X"), 1, j))
            s_Pass_list = sorted(list(set(Pass_list)))
            for ps in s_Pass_list:
              if "X" in ps and "Y" in ps:
                print(ps)
                

"""
実行結果：
X機 => コンピュータによる => 情報処理システムの => Yに関する
X機 => コンピュータによる => 情報処理システムの => 実現に関する => 研究分野とも => される
X機科学 => の
X機科学 => の => 一分野を => 指す
X機科学 => の => 一分野を => 指す => Y
X知能 => Y
X知能 => 語 => 研究分野とも => される
X行動を => 代わって => 行わせる
X行動を => 代わって => 行わせる => Yまたは
X解決などの => 知的行動を => 代わって => 行わせる
X解決などの => 知的行動を => 代わって => 行わせる => Yまたは
X解決などの => 知的行動を => 代わって => 行わせる => 技術または => 研究分野とも => される
X => Yとは
X => Y解決などの
X => という
X => という => 道具を => 用いて
X => という => 道具を => 用いて => Yする
X => という => 道具を => 用いて => 研究する => 計算機科学 => の
X => という => 道具を => 用いて => 研究する => 計算機科学 => の => 一分野を => 指す
X => という => 道具を => 用いて => 研究する => 計算機科学 => の => 一分野を => 指す => Y
X => エーアイとは => Y
X => エーアイとは => 語 => 研究分野とも => される
X => 問題解決などの => 知的行動を => 代わって => 行わせる
X => 問題解決などの => 知的行動を => 代わって => 行わせる => Yまたは
X => 問題解決などの => 知的行動を => 代わって => 行わせる => 技術または => 研究分野とも => される
X => 研究分野とも => される
Xこうちのう => Y
Xこうちのう => 語 => 研究分野とも => される
Xする => 計算機科学 => の
Xする => 計算機科学 => の => 一分野を => 指す
Xする => 計算機科学 => の => 一分野を => 指す => Y
Xする => 計算機科学 => の => 一分野を => 指す => 語 => 研究分野とも => される
Xと => 道具を => 用いて
Xと => 道具を => 用いて => Yする
Xと => 道具を => 用いて => 研究する => 計算機科学 => の
Xと => 道具を => 用いて => 研究する => 計算機科学 => の => 一分野を => 指す
Xと => 道具を => 用いて => 研究する => 計算機科学 => の => 一分野を => 指す => Y

"""