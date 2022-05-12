import re
import UK

basic_target = r'^\{\{基礎情報.*?$(.*?)^\}\}'
list_basic_target = re.findall(basic_target, UK.text_output("イギリス"), re.MULTILINE + re.DOTALL)

target = r'^\|(.*?)\s*=\s*(.*?)$'
for info in list_basic_target:
    list_target = re.findall(target, info, re.MULTILINE + re.DOTALL)

target_amari = r'(^\*.*?)^\|'
for country_name in list_basic_target:
    list_target_amari = re.findall(target_amari, country_name, re.MULTILINE + re.DOTALL)

dictionary = {}
for info in list_target:
    dictionary[info[0]] = info[1]

dictionary['公式国名'] += '\n'
for country_name in list_target_amari:
    dictionary['公式国名'] += country_name
#1度に抜き出せなかったため抜き出せなかった分の追加

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
outside_link = r'\[http.*?\]'
for i in dictionary.keys():
    dictionary[i] = re.sub(outside_link, '', dictionary[i])

#2重の{{の削除、整形
template1 = r'\{\{([^\}]*?)\|([^\}]*?)\|([^\}]*?)\|([^\}]*?)\|([^\}]*?)\}\}'
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
国旗画像 Flag of the United Kingdom.svg
国章画像 85px|イギリスの国章
国章リンク （国章）
標語 Dieu et mon droit（フランス語:神と我が権利）
国歌 God Save the Queen神よ女王を護り賜え
地図画像 Europe-UK.svg
位置画像 United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
公用語 英語
首都 ロンドン（事実上）

リーダブルコード:
変数をわかりやすくした
ごちゃごちゃしているので、コメントを入れてわかりやすくした。
'''

