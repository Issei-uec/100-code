import re
import UK

target = r'^\{\{基礎情報.*?$(.*?)^\}\}'
me = re.findall(target, UK.text_output("イギリス"), re.MULTILINE + re.DOTALL)
target1 = r'^\|(.*?)\s*=\s*(.*?)$'
target2 = r'(^\*.*?)^\|'
for i in me:
    me1 = re.findall(target1, i, re.MULTILINE + re.DOTALL)

for i in me:
    me2 = re.findall(target2, i, re.MULTILINE + re.DOTALL)

dictionary = {}
for i in me1:
    dictionary[i[0]] = i[1]

dictionary['公式国名'] += '\n'
for i in me2:
    dictionary['公式国名'] += i

dictionary['公式国名'] = dictionary['公式国名'].rstrip()


emphasize = r'\'{2,5}'

for i in dictionary.keys():
    dictionary[i] = re.sub(emphasize, "", dictionary[i])

for key, value in dictionary.items():
    print(key, value)