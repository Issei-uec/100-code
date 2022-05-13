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
略名 イギリス
日本語国名 グレートブリテン及び北アイルランド連合王国
公式国名 United Kingdom of Great Britain and Northern Ireland英語以外での正式国名:
An Rìoghachd Aonaichte na Breatainn Mhòr agus Eirinn mu Thuath（スコットランド・ゲール語）
Teyrnas Gyfunol Prydain Fawr a Gogledd Iwerddon（ウェールズ語）
Ríocht Aontaithe na Breataine Móire agus Tuaisceart na hÉireann（アイルランド語）
An Rywvaneth Unys a Vreten Veur hag Iwerdhon Glédh（コーンウォール語）
Unitit Kinrick o Great Breetain an Northren Ireland（スコットランド語）
Claught Kängrick o Docht Brätain an Norlin Airlann、Unitet Kängdom o Great Brittain an Norlin Airlann（アルスター・スコットランド 
語）
国旗画像 Flag of the United Kingdom.svg
国章画像 85px|イギリスの国章
国章リンク （国章）
標語 Dieu et mon droit（フランス語:神と我が権利）
国歌 God Save the Queen神よ女王を護り賜え
地図画像 Europe-UK.svg
位置画像 United Kingdom (+overseas territories) in the World (+Antarctica claims).svg
公用語 英語
首都 ロンドン（事実上）
最大都市 ロンドン
元首等肩書 女王
元首等氏名 エリザベス2世
首相等肩書 首相
首相等氏名 ボリス・ジョンソン
他元首等肩書1 貴族院議長
他元首等氏名1 ノーマン・ファウラー
他元首等肩書2 庶民院議長
他元首等氏名2 リンゼイ・ホイル
他元首等肩書3 最高裁判所長官
他元首等氏名3 ブレンダ・ヘイル
面積順位 76
面積大きさ 1 E11
面積値 244,820
水面積率 1.3%
人口統計年 2018
人口順位 22
人口大きさ 1 E7
人口値 6643万5600
人口密度値 271
GDP統計年元 2012
GDP値元 1兆5478億
GDP統計年MER 2012
GDP順位MER 6
GDP値MER 2兆4337億
GDP統計年 2012
GDP順位 6
GDP値 2兆3162億
GDP/人 36,727
建国形態 建国
確立形態1 イングランド王国／スコットランド王国（両国とも1707年合同法まで）
確立年月日1 927年／843年
確立形態2 グレートブリテン王国成立（1707年合同法）
確立年月日2 1707年5月1日
確立形態3 グレートブリテン及びアイルランド連合王国成立（1800年合同法）
確立年月日3 1801年1月1日
確立形態4 現在の国号「グレートブリテン及び北アイルランド連合王国」に変更
確立年月日4 1927年4月12日
通貨 UKポンド (£)
通貨コード GBP
時間帯 ±0
夏時間 +1
ISO 3166-1 GB / GBR
ccTLD .uk / .gb使用は.ukに比べ圧倒的少数。
国際電話番号 44
注記

リーダブルコード:
変数をわかりやすくした(p10)
ごちゃごちゃしているので、コメントを入れてわかりやすくした。(p76)
'''

