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

for key, value in dictionary.items():
    print(key, value)


'''
実行結果:
国旗画像 Flag of the United Kingdom.svg
国章画像 85px|イギリスの国章
国章リンク （国章）
標語 {{lang|fr|Dieu et mon droit}}<br />（フランス語:神と我が権利）
国歌 {{lang|en|God Save the Queen}}{{en icon}}<br />神よ女王を護り賜え<br />{{center|ファイル:United States Navy Band - God Save the Queen.ogg}}
地図画像 Europe-UK.svg
位置画像 United Kingdom (+overseas territories) in the World (+Antarctica claims).svg

リーダブルコード:
変数をわかりやすくした。(p10)
ごちゃごちゃしているので、コメントを入れてわかりやすくした。(p76)
'''
