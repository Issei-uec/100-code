import re
import UK

target = r'(\={2,4})\s*(.*?)\s*(\={2,4})'
list_target = re.findall(target, UK.text_output("イギリス"))
for i in range(len(list_target)):
    print(list_target[i][1] + ':' + str(len(list_target[i][0])-1))

'''
実行結果:
国名:1
歴史:1
地理:1
主要都市:2
気候:2
政治:1
元首:2

リーダブルコード:
変数をわかりやすくした
'''