f = open('popular-names.txt', 'r')

data = f.read()
'''
for i in data:
    if i == '\t':
        i = i.replace('\t', ' ')
'''
data1 = data.replace('\t', ' ')
print(data1)
f.close()