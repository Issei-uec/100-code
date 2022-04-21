m_file = open('popular-names.txt', 'r')

data = m_file.read()
data_word_split = data.split()
data_line_split = data.split('\n')

wordlist = []
for i in range(len(data_word_split)):
    if i % 4 == 2:
        wordlist.append(data_word_split[i])

sort_word_list = [wordlist[0]]
sort_line_list = [data_line_split[0]]

for i in range(len(wordlist)-1):
    for j in range(len(sort_word_list)):
        if int(wordlist[i+1]) > int(sort_word_list[j]):
            sort_word_list.insert(j, wordlist[i+1]) 
            sort_line_list.insert(j, data_line_split[i+1])
            break
        
        elif int(sort_word_list[j]) == int(sort_word_list[-1]):
            sort_word_list.append(wordlist[i+1])
            sort_line_list.append(data_line_split[i+1])
            break

#あまりキレイではない

txt_3 = open('col3.txt', 'w')
for i in sort_line_list:
    txt_3.write(i + '\n')
txt_3.close()

m_file.close()



"""
実行結果:
Linda	F	99689	1947
Linda	F	96211	1948
James	M	94757	1947
Michael	M	92704	1957
Robert	M	91640	1947

UNIX:sort -k 3 -t " " -r -n test.txt > 18.txt
Linda F 99689 1947
Linda F 96211 1948
James M 94757 1947
Michael M 92704 1957
Robert M 91640 1947

リーダブルコード:
変数の名前をわかりやすく
if elseの重要度
段落を気にした
"""