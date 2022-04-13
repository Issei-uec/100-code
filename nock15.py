str = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
erase_com = str.replace(',','')
erase_com_pir = erase_com.replace('.','')
split_str = erase_com_pir.split()

ans = []
order = 1
for i in split_str:
    if order==1 or order==5 or order==6 or order==7 or order==8 or order==9 or order==15 or order==16 or order==19:
        letter = i[0]
    else:
        letter = i[0:2]
    ans.append(letter)
    order = order+1

print(ans)