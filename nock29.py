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

#強調表現の削除
emphasize = r'\'{2,5}'
for i in dictionary.keys():
    dictionary[i] = re.sub(emphasize, "", dictionary[i])

#内部リンクの削除
inside_link1 = r'\[\[([^\]]*?)\|(.*?)\]\]'
for i in dictionary.keys():
    dictionary[i] = re.sub(inside_link1, r'\2', dictionary[i])

inside_link2 = r'\[\[(.*?)\]\]'
for i in dictionary.keys():
    dictionary[i] = re.sub(inside_link2, r'\1', dictionary[i])

#箇条書きの削除
kajou = r'\*{1,2}(.*?)'
for i in dictionary.keys():
    dictionary[i] = re.sub(kajou, r'\1', dictionary[i])

#htmlタグの削除
html_tag = r'\<.*?\>'
for i in dictionary.keys():
    dictionary[i] = re.sub(html_tag, '', dictionary[i])

#外部リンクの削除
outside_link = r'\[http(.*?)\]'
for i in dictionary.keys():
    dictionary[i] = re.sub(outside_link, '', dictionary[i])

#2重の{{の削除、整形
template1 = r'\{\{([^\}]*?)\|(.*?)\|([^\}]*?)\|([^\}]*?)\|([^\}]*?)\}\}'
for i in dictionary.keys():
    dictionary[i] = re.sub(template1, '', dictionary[i])

template2 = r'\{\{([^\}]*?)\|([^\}]*?)\|([^\}]*?)\|([^\}]*?)\}\}'
for i in dictionary.keys():
    dictionary[i] = re.sub(template2, r'\2', dictionary[i])

template3 = r'\{\{(.*?)\|(.*?)\|(.*?)\}\}'
for i in dictionary.keys():
    dictionary[i] = re.sub(template3, r'\3', dictionary[i])

template4 = r'\{\{(.*?)\}\}'
for i in dictionary.keys():
    dictionary[i] = re.sub(template4, '', dictionary[i])

for key, value in dictionary.items():
    print(key, value)

'''
実行結果:



'''

