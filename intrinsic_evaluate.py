import sys

from smart_open import open
import json

from evaluate import get_score
from configs import OUT, POS, TEST, METHOD, MODE, EMB_PATH, FILTER_1, FILTER_2

if TEST == 'codalab':
    print('You have been using the codalab test set, you dont have laballed data for internal evaluation!')
    sys.exit()
    
## {'ОТЕЧЕСТВО': ['445-N', '445-N', '130809-N', '445-N', '144422-N'], 'ТОММОТ': ['242-N', '142582-N', '145516-N']}
gold_dict = json.load(open('%sgold_dicts/%s_%s_%s_gold.json' % (OUT, POS, TEST, METHOD), 'r'))
## PRED: {'АБДОМИНОПЛАСТИКА': ['100022-N', '242-N', '2062-N', '106555-N', '2550-N', '139862-N', '106451-N', ...
pred_dict = json.load(open('%sresults/%s_%s_%s_%s_%s_pred.json' % (OUT, POS, TEST, METHOD, FILTER_1, FILTER_2), 'r'))

first2pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:2]}
print('GOLD ', first2pairs_gold)
first2pairs_pred = {k: pred_dict[k] for k in list(pred_dict)[:2]}
print('PRED ', first2pairs_pred)

print(len(gold_dict), len(pred_dict))

print('Embedding:', EMB_PATH.split('/')[-1], file=sys.stderr)
print(TEST, METHOD, MODE, FILTER_1, FILTER_2)
mean_ap, mean_rr = get_score(gold_dict, pred_dict)
print("MAP: {0}\nMRR: {1}\n".format(mean_ap, mean_rr), file=sys.stderr)

