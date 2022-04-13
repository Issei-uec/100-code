def ngram(n, strn):
    return list(zip(*[strn[i:] for i in range(n)]))

str = 'I am an NLPer'
erase_com = str.replace(',','')
erase_com_pir = erase_com.replace('.','')
split_str = erase_com_pir.split()

moji_bi = ngram(2, str)
tng_bi = ngram(2, split_str)

print(tng_bi)
print(moji_bi)
