def ngram(n, strn):
    main_list = []
    for i in range(len(strn)-n+1):
        sub_list = []
        for j in range(n):
            sub_list.append(strn[i+j])
        main_list.append(sub_list)
        main_list[i] = tuple(main_list[i])     
    return main_list 
 
 #もっとコンパクトに書けると思う
    
    

str = 'I am an NLPer'
erase_com = str.replace(',','')
erase_com_pir = erase_com.replace('.','')
split_str = erase_com_pir.split()

#split_str = str.split()

tng_bi = ngram(2, split_str)
moji_bi = ngram(2, str)

print('単語bigram:', tng_bi)
print('文字bigram:', moji_bi)

"""
実行結果：
単語bigram: [('I', 'am'), ('am', 'an'), ('an', 'NLPer')]
文字bigram: [('I', ' '), (' ', 'a'), ('a', 'm'), ('m', ' '), (' ', 'a'), ('a', 'n'), ('n', ' '), (' ', 'N'), ('N', 'L'), ('L', 'P'), 
('P', 'e'), ('e', 'r')]
"""
