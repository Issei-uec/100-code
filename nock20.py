import random

def typoglycemia(str):
    shuffle = []
    for i in str.split():
        if len(i) > 4:
            i = i[:1] + ''.join(random.sample(i[1:-1], len(i)-2)) + i[-1:]
        shuffle.append(i)
    return ' '.join(shuffle)


X = input('str:')
print(typoglycemia(X))