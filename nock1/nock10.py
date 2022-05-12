import random

def typoglycemia(str):
    shuffle = []
    for word in str.split():
        if len(word) > 4:
            randomword = ''.join(random.sample(word[1:-1], len(word[1:-1])))
            word = word[:1] + randomword + word[-1:]
        shuffle.append(word)
        
    return ' '.join(shuffle)


X = input('原文: ')
print('shuffle:', typoglycemia(X))

"""
実行結果
原文: I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind . 
shuffle: I c'ouldnt bveleie that I cluod alctulay unnesradtd what I was rnaedig : the phomneenal power of the huamn mind .
"""