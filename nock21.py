f = open('popular-names.txt', 'r')

data = f.read()
dannraku = []
for i in data:
    if i == '\n':
        dannraku += 1

print(dannraku)

f.close()

