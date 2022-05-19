import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

word_list = []
dic_word = {}

for word in list:
    if word["surface"] != "。" and word["surface"] != "、" and word["surface"] != "「" and word["surface"] != "」":
        if not word["surface"] in dic_word:
            dic_word[word["surface"]] = 1
        else:
            dic_word[word["surface"]] += 1

sort_dic_word = sorted(dic_word.items(), key=lambda x:x[1], reverse=True)
print(sort_dic_word)

"""
実行結果：
[('の', 9103), ('て', 6697), ('は', 6384), ('に', 6148), ('を', 6068), ('と', 5474), ('が', 5258), ('た', 3916), ('で', 3783), ('
も', 2433), ('だ', 2270), ('し', 2264), ('ない', 2254), ('から', 2001), ('ある', 1714), ('な', 1579), ('か', 1432), ('ん', 1415), 
('いる', 1249), ('事', 1177), ('へ', 1033), ('する', 986), ('もの', 972), ('です', 960), ('君', 953), ('云う', 937), ('主人', 928), ('う', 922), ('よう', 687), ('ね', 673), ('この', 635)

リーダブルコード：

"""