txt_1 = open('col1.txt', 'r')
txt_2 = open('col2.txt', 'r')
txt_1_2 = open('col_12.txt', 'w')

data1 = txt_1.read().split()
data2 = txt_2.read().split()

for i, j in zip(data1, data2):
    txt_1_2.write(i + '\t' + j + '\n')

txt_1_2.close()
txt_1.close()
txt_2.close()


"""
実行結果:
Mary	F
Anna	F
Emma	F
Elizabeth	F
Minnie	F

UNIX:paste -d "\t" col1.txt col2.txt > col1-2.txt

リーダブルコード:
変数の名前をわかりやすく
段落を気にした
"""