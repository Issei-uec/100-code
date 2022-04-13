def ngram(n, strn):
    return list(zip(*[strn[i:] for i in range(n)]))

str = 'I am an NLPer'
str1 = str.replace(',','')
str2 = str1.replace('.','')
tng = str2.split()

tng1 = ngram(2, str)
tng2 = ngram(2, tng)

print(tng1)
print(tng2)
