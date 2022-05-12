import re
import UK

target = r'(\[\[Category:.*\]\])'
me = re.findall(target, UK.text_output("イギリス"))

for category in me:
    print(category)

'''
実行結果:



'''