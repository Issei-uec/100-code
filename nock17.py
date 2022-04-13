def ngram(n, strn):
    return list(zip(*[strn[i:] for i in range(n)]))

str1 = 'paraparaparadise'
str2 = 'paragraph'

str3 = ngram(2, str1)
str4 = ngram(2, str2)

print('X = ', ngram(2, str1))

print('Y = ', ngram(2, str2))

print('和集合：', set(str3) | set(str4))
print('積集合：', set(str3) & set(str4))
print('差集合：', set(str3) - set(str4))

Z = {('s', 'e')}
print('Xにseが含まれるか:', Z <= set(str3))
print('Yにseが含まれるか:', Z <= set(str4))