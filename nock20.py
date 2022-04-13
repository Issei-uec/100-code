import random

def typoglycemia(str):
    shuffle = []
    for word in str.split():
        if len(word) > 4:
            randomword = ''.join(random.sample(word[1:-1], len(word[1:-1])))
            word = word[:1] + randomword + word[-1:]
        shuffle.append(word)
        
    return ' '.join(shuffle)


X = input('str:')
print('shuffle:', typoglycemia(X))