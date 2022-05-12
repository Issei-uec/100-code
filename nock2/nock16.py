m_file = open('popular-names.txt', 'r')

data = m_file.read()
data_split = data.split('\n')
del data_split[-1]

N = int(input('行数:'))
for paragraph in range(N):
    print(data_split[-N+paragraph])

m_file.close()


"""
実行結果:
行数:5
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018

UNIX:tail -n 5 test.txt

リーダブルコード:
変数の名前をわかりやすく
段落を気にした
"""