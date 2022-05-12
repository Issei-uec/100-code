import re
import UK

target = r'(\[\[Category:.*\]\])'
list_target = re.findall(target, UK.text_output("イギリス"))

for category in list_target:
    print(category)

'''
実行結果:
[[Category:イギリス|*]]
[[Category:イギリス連邦加盟国]]
[[Category:英連邦王国|*]]
[[Category:G8加盟国]]
[[Category:欧州連合加盟国|元]]
[[Category:海洋国家]]
[[Category:現存する君主国]]
[[Category:島国]]
[[Category:1801年に成立した国家・領域]]

リーダブルコード:
変数をわかりやすくした
'''