import requests
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

URL = "https://en.wikipedia.org/w/api.php"

asa = str(re.sub(r"\s", "_", dictionary["国旗画像"]))

PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": f"File:{asa}",
    "iiprop": "url"
}

URL += "?"
for key, value in PARAMS.items():
    URL += "&" + key + "=" +  value

url = requests.get(URL)
print(re.findall(r'\"url\"\:\"(.*?)\"', url.text))
    
'''
実行結果:
['https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg']

リーダブルコード:
変数をわかりやすくした(p10)
'''