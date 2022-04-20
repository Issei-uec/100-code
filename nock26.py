f = open('popular-names.txt', 'r')
N = int(input('行数:'))

data = f.read()
data_split = data.split('\n')

for i in range(N):
    print(data_split[-N+i-1])

f.close()