import re
import UK

target = r'(\={2,4})\s*(.*?)\s*(\={2,4})'
me = re.findall(target, UK.text_output("イギリス"))

for i in range(len(me)):
    print(me[i][1] + ':' + str(len(me[i][0])-1))