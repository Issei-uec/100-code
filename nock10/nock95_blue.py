from sacrebleu.metrics import BLEU


with open('/home2/y2019/o1910142/kftt-data-1.0/data/orig/kyoto-test.en', 'r') as f:
#with open('/home2/y2019/o1910142/kftt-data-1.0/data/orig/kyoto-dev.en', 'r') as f:
    lines = [x.strip() for x in f]

with open('/home2/y2019/o1910142/BLEU_test_95_t.txt', 'r') as f:
#with open('/home2/y2019/o1910142/BLEU_test_95.txt', 'r') as f:
    trans = [x.strip() for x in f]
"""with open('/home2/y2019/o1910142/BLEU_test_98.txt', 'r') as f:
    trans = [x.strip() for x in f]"""

base = lines[0:1000]
trans = trans[0:1000]

bleu = BLEU()
result = bleu.corpus_score(trans, [base])
print(result)

"""
BLEU = 9.01 51.7/20.6/10.1/5.4 (BP = 0.581 ratio = 0.648 hyp_len = 14495 ref_len = 22367) #dev
BLEU = 11.87 53.9/24.0/13.3/8.3 (BP = 0.611 ratio = 0.670 hyp_len = 15168 ref_len = 22644) #test
"""