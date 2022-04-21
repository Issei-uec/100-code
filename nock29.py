m_file = open('popular-names.txt', 'r')

data = m_file.read()
data_split = data.split()
data_para_split = data.split('\n')

wordlist = []
for i in range(len(data_split)):
    if i % 4 == 2:
        wordlist.append(data_split[i])

sort_list = [wordlist[0]]
sort_p_list = [data_para_split[0]]


for i in range(len(wordlist)-1):
    for j in range(len(sort_list)):
        if int(wordlist[i+1]) > int(sort_list[j]):
            sort_list.insert(j, wordlist[i+1]) 
            sort_p_list.insert(j, data_para_split[i+1])
            break
        
        elif int(sort_list[j]) == int(sort_list[-1]):
            sort_list.append(wordlist[i+1])
            sort_p_list.append(data_para_split[i+1])
            break


txt_3 = open('col3.txt', 'w')
for i in sort_p_list:
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
"""