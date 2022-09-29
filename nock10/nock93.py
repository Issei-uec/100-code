from sacrebleu.metrics import BLEU

with open('/home2/y2019/o1910142/kftt-data-1.0/data/orig/kyoto-dev.en', 'r') as f:
    lines = [x.strip() for x in f]

with open('/home2/y2019/o1910142/BLEU_test.txt', 'r') as f:
    trans = [x.strip() for x in f]
"""with open('/home2/y2019/o1910142/BLEU_test_98.txt', 'r') as f:
    trans = [x.strip() for x in f]"""

base = lines[0:300]
trans = trans[0:300]

bleu = BLEU()
result = bleu.corpus_score(trans, [base])
print(result)

"""
BLEU = 8.66 43.9/15.0/6.4/3.3 (BP = 0.800 ratio = 0.817 hyp_len = 6565 ref_len = 8033)
"""

