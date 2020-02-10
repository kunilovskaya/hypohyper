import sys

from smart_open import open
import json

from evaluate import get_score
from configs import OUT, POS, TEST, METHOD

if TEST == 'provided':
    print('You have been useing the provided test set, you dont have laballed data for internal evaluation!')
    sys.exit()

gold_dict = json.load(open('%sgold_dicts/%s_%s_gold.json' % (OUT, POS, TEST), 'r'))
pred_dict = json.load(open('%sresults/%s_%s_pred_%s.json' % (OUT, POS, TEST, METHOD), 'r'))

first2pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:2]}
print(first2pairs_gold)
first2pairs_gold_pred = {k: pred_dict[k] for k in list(pred_dict)[:2]}
print(first2pairs_gold_pred)


mean_ap, mean_rr = get_score(gold_dict, pred_dict)
print("MAP: {0}\nMRR: {1}\n".format(mean_ap, mean_rr), file=sys.stderr)

