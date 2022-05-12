import re
import UK

target = r'Category:(.*)\]\]'
me = re.findall(target, UK.text_output("イギリス"))

for category_name in me:
    print(category_name)