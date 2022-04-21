f = open('popular-names.txt', 'r')
N = int(input('行数:'))

data = f.read()
data_split = data.split('\n')

for i in range(N):
    print(data_split[-N+i-1])

f.close()


"""
実行結果:
行数:5
Benjamin        M       13381   2018
Elijah  M       12886   2018
Lucas   M       12585   2018
Mason   M       12435   2018
Logan   M       12352   2018

UNIX:tail -n 5 test.txt

"""