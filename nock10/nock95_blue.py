from sacrebleu.metrics import BLEU


with open('/home2/y2019/o1910142/kftt-data-1.0/data/orig/kyoto-dev.en', 'r') as f:
    lines = [x.strip() for x in f]

with open('/home2/y2019/o1910142/BLEU_test_95.txt', 'r') as f:
    trans = [x.strip() for x in f]
"""with open('/home2/y2019/o1910142/BLEU_test_98.txt', 'r') as f:
    trans = [x.strip() for x in f]"""

base = lines[0:300]
trans = trans[0:300]

bleu = BLEU()
result = bleu.corpus_score(trans, [base])
print(result)

"""
dev_data: BLEU = 7.82 51.6/19.3/9.0/4.8 (BP = 0.542 ratio = 0.620 hyp_len = 4983 ref_len = 8033)
"""