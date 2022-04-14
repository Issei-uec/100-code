str = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
erase_com = str.replace(',','')
erase_com_pir = erase_com.replace('.','')
split_str = erase_com_pir.split()

ans = []
number = 1
for i in split_str:
    if number==1 or number==5 or number==6 or number==7 or number==8 or number==9 or number==15 or number==16 or number==19:
        elem = i[0]
    else:
        elem = i[0:2]

    ans.append(elem)
    number = number+1

print(ans)

#実行結果：['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mi', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca']