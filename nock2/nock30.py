import pandas as pd
datafile = pd.read_table('popular-names.txt', sep="\t", header=None)

frequency = datafile[0].value_counts()
print(frequency)
print(datafile)
"""
実行結果:
James      118
William    111
Robert     108
John       108
Mary        92

UNIX:
cut -f 1 -d " " test.txt > 19.txt
sort 19.txt > 19_sort.txt
uniq -c 19_sort.txt > 19_sort_uniq.txt
sort -k 1 -t " " -r -n 19_sort_uniq.txt > 19_fin.txt
    118 James
    111 William
    108 Robert
    108 John
     92 Mary

リーダブルコード:
変数の名前をわかりやすく
"""