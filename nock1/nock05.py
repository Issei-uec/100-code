str = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
erase_com = str.replace(',','')
erase_com_pir = erase_com.replace('.','')
split_str = erase_com_pir.split()

ans = []
dict01 = {}
number1 = [1, 5, 6, 7, 8, 9, 15, 16, 19]
number = 1
for i in split_str:
    if number in number1:
        dict01[i[0]] = number
    else:
        dict01[i[0:2]] = number
    number = number+1

print(dict01)

#実行結果：['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mi', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca']