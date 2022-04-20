f = open('popular-names.txt', 'r')

data = f.read()
data1 = data.replace('\t', ' ')
print(data1)
f.close()