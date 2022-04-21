from pprint import pprint
import pandas as pd

m_file = open('popular-names.txt', 'r')

data = m_file.read()
data1 = data.replace('\t', ' ')
print(data1)


m_file.close()

"""
実行結果:
Mary F 7065 1880
Anna F 2604 1880
Emma F 2003 1880
Elizabeth F 1939 1880
Minnie F 1746 1880

UNIX:tr "\t" " " <popular-names.txt> test.txt

"""