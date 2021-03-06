import sys

from smart_open import open
import json

from evaluate import get_score
from trials_errors.configs import OUT, POS, TEST, METHOD, MODE, EMB_PATH, FILTER_1, FILTER_2

if 'codalab' in TEST:
    print('You have been using the codalab test set, you dont have laballed data for internal evaluation!')
    sys.exit()
    
## # {'WORD1': [['4544-N'], ['147272-N']], 'WORD2': [['141697-N', '116284-N']]}
gold_dict = json.load(open('../gold_dicts/%s_%s_%s_gold.json' % (POS, TEST, METHOD), 'r'))
## PRED: {'АБДОМИНОПЛАСТИКА': ['100022-N', '242-N', '2062-N', '106555-N', '2550-N', '139862-N', '106451-N', ...
pred_dict = json.load(open('%sresults/org_split/%s_%s_%s_%s_%s_pred.json' % (OUT, POS, TEST, METHOD, FILTER_1, FILTER_2), 'r'))

first2pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:2]}
print('GOLD ', first2pairs_gold)
first2pairs_pred = {k: pred_dict[k] for k in list(pred_dict)[:2]}
print('PRED ', first2pairs_pred)

print(len(gold_dict), len(pred_dict))

print('Embedding:', EMB_PATH.split('/')[-1], file=sys.stderr)
print(TEST, METHOD, MODE, FILTER_1, FILTER_2)
mean_ap, mean_rr = get_score(gold_dict, pred_dict)
print("MAP: {0}\nMRR: {1}\n".format(mean_ap, mean_rr), file=sys.stderr)

# for k, v in gold_dict.items():
#     if k in ['ЧИБИС', 'ШАШКА', 'ШАТТЛ', 'ШАТУН', 'ЧАСТУШКА']:
#         print(k,v)

