word_list2 = []
dic_word2 = {}
for word in list:
    if word["surface"] != "、" and word["surface"] != "」" and word["surface"] != "「":
        word_list2.append(word["surface"])

for word in list:
    if word["surface"] != "。" and word["surface"] != "、" and word["surface"] != "」" and word["surface"] != "「":
        if not word["surface"] in dic_word2:
            dic_word2[word["surface"]] = 0
flag = 0
for i in range(len(word_list2)):
    if word_list2[i] == "猫":
        flag = 1
    elif word_list2[i] == "。":
        flag = 0
    elif flag == 1:
        dic_word2[word_list2[i]] += 1

for i in range(len(word_list2)):
    if word_list2[-i] == "猫":
        flag = 1
    elif word_list2[-i] == "。":
        flag = 0
    elif flag == 1:
        dic_word2[word_list2[-i]] += 1
    

print(dic_word2)





sort_dic_word = sorted(dic_word2.items(), key=lambda x:x[1], reverse=True)