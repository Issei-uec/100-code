m_file = open('popular-names.txt', 'r')

data_tab = m_file.read()
data_space = data_tab.replace('\t', ' ')

print(data_space)
m_file.close()

"""
実行結果:
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880

UNIX:tr "\t" " " <popular-names.txt> test.txt

リーダブルコード:
変数の名前をわかりやすく
段落を気にした
"""