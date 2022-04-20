f = open('popular-names.txt', 'r')

data = f.read()
dannraku = 0
for i in data:
    if i == '\n':
        dannraku += 1

print(dannraku)

f.close()

