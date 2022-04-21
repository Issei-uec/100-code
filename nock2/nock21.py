m_file = open('popular-names.txt', 'r')
data = m_file.read()

paragraph = 0
for i in data:
    if i == '\n':
        paragraph += 1

print(paragraph)
m_file.close()


"""
実行結果:2780
UNIX: wc -l popular-names.txt
リーダブルコード:
変数の名前をわかりやすく
段落を気にした
"""