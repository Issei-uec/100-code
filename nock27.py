m_file = open('popular-names.txt', 'r')
data = m_file.read()
split = data.split('\n')

paragraph = 0
for i in data:
    if i == '\n':
        paragraph += 1

N = input('ファイル分割数:')

for i in range(int(N)-1):
    txt = open('split' + str(i) + '.txt', 'w')
    for j in range(paragraph//int(N)):       
        txt.write(split[(paragraph//int(N))*i + j] + '\n')

    txt.close()

txt = open('split' + str(int(N)-1) + '.txt', 'w')
for j in range(paragraph//int(N) + paragraph%int(N)):       
    txt.write(split[(paragraph//int(N))*(int(N)-1) + j] + '\n')

txt.close()

m_file.close()



"""
実行結果:

UNIX: split -n 3 test.txt
xaa
xab
xac
"""