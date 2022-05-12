import re
import UK

target = r'ファイル:(.*?)\|'
me = re.findall(target, UK.text_output("イギリス"))

for file in me:
    print(file)