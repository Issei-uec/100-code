import re
import UK

target = r'Category:(.*)\]\]'
list_target = re.findall(target, UK.text_output("イギリス"))

for category_name in list_target:
    print(category_name)

'''
実行結果:
イギリス|*
イギリス連邦加盟国
英連邦王国|*
G8加盟国
欧州連合加盟国|元
海洋国家
現存する君主国
島国
1801年に成立した国家・領域

リーダブルコード:
変数をわかりやすくした
'''