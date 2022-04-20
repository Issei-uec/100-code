def ngram(n, strn):
    main_list = []
    for i in range(len(strn)-n+1):
        sub_list = []
        for j in range(n):
            sub_list.append(strn[i+j])
        main_list.append(sub_list)     
    return main_list 

#ngramはできたが、表示方法がリストの中にリストとなっていて見にくいため、改善の余地あり
    
    

str = 'I am an NLPer'
erase_com = str.replace(',','')
erase_com_pir = erase_com.replace('.','')
split_str = erase_com_pir.split()

moji_bi = ngram(2, str)
tng_bi = ngram(2, split_str)

print(tng_bi)
print(moji_bi)
