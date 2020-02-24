##  json вот с таким содержанием [('['усмешка_NOUN', 'ухмылка_NOUN'], ['преподнесение_NOUN', 'представление_NOUN'])]

import argparse
import sys, os
import time
from hyper_imports import wd2id_dict, id2wds_dict, read_xml, get_orgtest, get_orgtrain
from hyper_imports import load_embeddings, preprocess_wordpair
from configs import VECTORS, EMB_PATH, OUT, POS, TAGS, MWE, SKIP_OOV, METHOD, TEST, RUWORDNET
import pandas as pd
import json
from collections import defaultdict


def convert(data, input):
    data_hyper_ids = defaultdict(list)
    
    for i in data:
        res = i.split('\t')
        twords, hyper_ids = res
        data_hyper_ids[hyper_ids].append(twords)
    out = defaultdict(list)
    
    for str_ids in temp:  # keys from source; vals are parents wordlists
        if str_ids in data_hyper_ids:  # keys from train; vals are twords
            
            out[str(data_hyper_ids[str_ids])].append(temp[str_ids])
    out2 = []
    for k, v in out.items():
        v = [item for sublist in v for item in sublist]  # flatten the list of hypernyms
        k = k.replace("'","").replace('[', '').replace(']', '')
        k = [i.strip() for i in k.split(', ')]  # get a list of hyponyms
        out2.append((k, v))
    print(out2[:1])
    
    return out2


parser = argparse.ArgumentParser()
if TEST == 'provided':
    parser.add_argument('--source', default='input/data/training_nouns.tsv', type=os.path.abspath,
                        help='train data in format SYNSET<TAB>SENSES<TAB>PARENTS<TAB>DEFINITION')
    if POS == 'NOUN':
        parser.add_argument('--train', default='input/data/org_split/train_nouns.tsv', type=os.path.abspath)
        parser.add_argument('--dev', default='input/data/org_split/dev_nouns.tsv', type=os.path.abspath)

    if POS == 'VERB':
        parser.add_argument('--train', default='input/data/org_split/train_verbs.tsv', type=os.path.abspath)
        parser.add_argument('--dev', default='input/data/org_split/dev_verbs.tsv', type=os.path.abspath)

else:
    print('Select "provided" for TEST in configs.py (train-dev-test split)', file=sys.stderr)
    
args = parser.parse_args()

source = [i.strip() for i in open(args.source, 'r').readlines()[1:]]

temp = {}
for i in source:
    res = i.split('\t')
    _, wds, par_ids, par_wds = res
    ## get gid of double quotes inside strings: ПРЕМИЯ "ОСКАР"
    par_wds = par_wds.replace('"', '')
    par_wds = par_wds.replace("'", '"')
    par_wds = json.loads(par_wds)

    temp[par_ids] = par_wds

train = [i.strip() for i in open(args.train, 'r').readlines()]
dev = [i.strip() for i in open(args.dev, 'r').readlines()]

OUT_TRAIN = '%strains/' % OUT
os.makedirs(OUT_TRAIN, exist_ok=True)

out_train = convert(train, temp)
print(len(out_train))
print()
out_dev = convert(dev, temp)
print(len(out_dev))

json.dump(out_train, open('%strain_%s_%s_%s.json' % (OUT_TRAIN, POS, TEST, METHOD), 'w'))
json.dump(out_dev, open('%sdev_%s_%s_%s.json' % (OUT_TRAIN, POS, TEST, METHOD), 'w'))