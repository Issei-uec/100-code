str = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'

erase_com = str.replace(',','')
erase_com_pir = erase_com.replace('.','')
split_str = erase_com_pir.split()

ans = []
for word in split_str:
    ans.append(len(word))
    
print(ans)

#実行結果：[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9]