f = open('popular-names.txt', 'r')
N = int(input('行数:'))

for i in range(N):
    print(f.readline())

f.close()