## look at lists of golden and returned lists: are my predicted words within the first 50?
import os
import sys
from smart_open import open
from collections import defaultdict
import json

from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, TAGS, \
    TEST, METHOD, FILTER_1, FILTER_2
from hyper_imports import id2wds_dict, read_xml
from hyper_imports import map_mwe, new_preprocess_mwe

    
# print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
# model = load_embeddings(EMB_PATH)

if POS == 'NOUN':
    synsets = '%ssynsets.N.xml' % RUWORDNET
elif POS == 'VERB':
    synsets = '%ssynsets.V.xml' % RUWORDNET
else:
    synsets = None
    
parsed_syns = read_xml(synsets)
id2lemmas = id2wds_dict(parsed_syns)

if VECTORS  == 'mwe-pos-vectors' and POS == 'NOUN':
    source = open('lists/ruWordNet_names.txt', 'r').readlines()
    source_tagged = open('lists/ruWordNet_same-names_pos.txt', 'r').readlines()
    mwe_map = map_mwe(names=source, same_names=source_tagged, tags=TAGS, pos=POS)
else:
    mwe_map = None

pred_lemid = json.load(open('errors/preds_lemid_%s_%s_%s_%s_%s.json' % (POS, TEST, METHOD, FILTER_1, FILTER_2), 'r'))
oov = []
gold_dict = json.load(open('gold_dicts/gold_%s_%s_%s.json' % (POS, TEST, METHOD), 'r'))
print('\nHuman-readable results for top ten test words:', file=sys.stderr)
for hypo, hyp_ids in gold_dict.items(): # {'WORD1': [['4544-N'], ['147272-N']], 'WORD2': [['141697-N', '116284-N']]}
    hyp_wds = defaultdict(list)
    for hyp_id_list in hyp_ids:
        this_comp = []
        for hyp_id in hyp_id_list:
            for w in id2lemmas[hyp_id]:
                w, _ = new_preprocess_mwe(w, pos=POS, tags=TAGS, map_mwe_names=mwe_map)
                this_comp.append(w+'_'+hyp_id)
        hyp_wds[hypo.strip()].append(this_comp)
    print()
    print(hypo, file=sys.stderr)
    try: #skipping OOV
        print('PRED:', pred_lemid[hypo.strip()], file=sys.stderr)
        print()
        print('GOLD:', hyp_wds[hypo.strip()], file=sys.stderr)
    except KeyError:
        print('OOOOVVVV', hypo)
        oov.append(hypo)
        continue
    


print(VECTORS, MODE,OOV_STRATEGY, TEST, METHOD, FILTER_1, FILTER_2, file=sys.stderr)
print('OOOVV', oov)