import re

neko = open('neko.txt.mecab', 'r', encoding="utf-8")
list = []
for line in neko:
    if line != "EOS\n":
        nekon = re.split(',|\t', line)
        list.append({"surface": nekon[0], "base": nekon[7], "pos": nekon[1], "pos1": nekon[2]})
neko.close()

N_in = []
no_in = []
for i in list:
    if len(N_in) != 0:
        if i["pos"] == "名詞":        
            if len(no_in) != 0:
                print(N_in[0] + "の" + i["surface"])
                N_in = []
                no_in = []
        elif i["surface"] == "の":
            no_in = [i["surface"]]
        else: 
            N_in = []
            no_in = []
    elif i["pos"] == "名詞":
        N_in = [i["surface"]]

    
"""
実行結果：
坊主の慣用
一所不住の沙門
彼等の仲間
彼等の肩
種の逆上
発明の売薬
彼等のため
蒲鉾の種
観音の像
八分の朽木
鴨南蛮の材料
下宿の牛鍋
臨時の気違
臨時の気違
一の狂人
学者の頭脳
湯の中
人の説
葡萄の湯
古人の真似
人の態度
酔っぱらいのよう
酒飲みのよう
線香の間
有名の大家
快走の上
文章の趣向
今日のところ
不可能の事
時機の到来

リーダブルコード：

"""
