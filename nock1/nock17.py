def ngram(n, strn):
    return list(zip(*[strn[i:] for i in range(n)]))

str1 = 'paraparaparadise'
str2 = 'paragraph'

str1_ngram = ngram(2, str1)
str2_ngram = ngram(2, str2)

print('X = ', ngram(2, str1))
print('Y = ', ngram(2, str2))

print('和集合：', set(str1_ngram) | set(str2_ngram))
print('積集合：', set(str1_ngram) & set(str2_ngram))
print('差集合：', set(str1_ngram) - set(str2_ngram))

Z = {('s', 'e')}
print('seがXに含まれるか:', Z <= set(str1_ngram))
print('seがYに含まれるか:', Z <= set(str2_ngram))