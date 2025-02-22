f = open('popular-names.txt', 'r')
txt_1 = open('col1.txt', 'w')
txt_2 = open('col2.txt', 'w')

data = f.read()
data_split = data.split()
for elem in range(len(data_split)):
    if elem % 4 == 0:
        txt_1.write(data_split[elem] + '\n')
    if elem % 4 == 1:
        txt_2.write(data_split[elem] + '\n')

txt_1.close()
txt_2.close()
f.close()

"""
実行結果:
col1:
Mary
Anna
Emma
Elizabeth
Minnie

col2:
F
F
F
F
F

UNIX:
cut -f 1 -d " " test.txt > col1.txt
cut -f 2 -d " " test.txt > col2.txt

リーダブルコード:
変数の名前をわかりやすく
段落を気にした
"""