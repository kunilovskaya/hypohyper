import sys
from smart_open import open
import json
from evaluate import get_score

# {'WORD1': [['4544-N'], ['147272-N']], 'WORD2': [['141697-N', '116284-N']]}
gold_dict = json.load(open(sys.argv[1], 'r'))
# PRED: {'АБДОМИНОПЛАСТИКА':
# ['100022-N', '242-N', '2062-N', '106555-N', '2550-N', '139862-N', '106451-N', ...
pred_dict = json.load(open(sys.argv[2], 'r'))

first2pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:2]}
print('GOLD ', first2pairs_gold)
first2pairs_pred = {k: pred_dict[k] for k in list(pred_dict)[:2]}
print('PRED ', first2pairs_pred)

print(len(gold_dict), len(pred_dict))

mean_ap, mean_rr = get_score(gold_dict, pred_dict)
print("MAP: {0:.4f}\nMRR: {1:.4f}\n".format(mean_ap, mean_rr))

# for k, v in gold_dict.items():
#     if k in ['ЧИБИС', 'ШАШКА', 'ШАТТЛ', 'ШАТУН', 'ЧАСТУШКА']:
#         print(k,v)
