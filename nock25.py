f = open('popular-names.txt', 'r')
N = int(input('行数:'))

for i in range(N):
    print(f.readline().replace("\n", ""))

f.close()


"""
実行結果:
行数:5
Mary    F       7065    1880
Anna    F       2604    1880
Emma    F       2003    1880
Elizabeth       F       1939    1880
Minnie  F       1746    1880

UNIX:head -n 5 test.txt

"""