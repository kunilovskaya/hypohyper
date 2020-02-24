##  json вот с таким содержанием [('['усмешка_NOUN', 'ухмылка_NOUN'], ['преподнесение_NOUN', 'представление_NOUN'])]

import argparse
import sys, os
import time
from hyper_imports import preprocess_mwe
from hyper_imports import load_embeddings, preprocess_wordpair
from configs import VECTORS, EMB_PATH, OUT, POS, TAGS, MWE, SKIP_OOV, METHOD, TEST, RUWORDNET
import pandas as pd
import json
from collections import defaultdict
from smart_open import open

def split_n_tag(bad_list): ## list in, list out
    res_list = []
    for i in bad_list:
        if ', ' not in i:
           i = preprocess_mwe(i, tags=TAGS, pos=POS)
           res_list.append(i)
        elif ', ' in i:
            items = [x.strip() for x in i.split(', ')]  # get a list of items in the string
            # flatter = [item for sublist in items for item in sublist]
            for ii in items:
                # print(type(ii))
                ii = preprocess_mwe(ii, tags=TAGS, pos=POS)
                res_list.append(ii)
                
    return res_list

def convert(data, input):
    # data_hyper_ids = defaultdict(list)
    twords = []
    for i in data:
        res = i.split('\t')
        tword, hyper_ids = res
        twords.append(tword)
        # data_hyper_ids[hyper_ids].append(tword)
        
    out = defaultdict(set)
    
    
    for wds in input:  # can be a string of several hyponyms from source; vals are parents wordlists
        wds_list = [i.strip() for i in wds.split(', ')]  # get a list of hyponyms
        # if 'КОРРЕСПОНДЕНТСКИЙ ПУНКТ' in wds_list:
        #     print('GOTCHA')
        for tw in twords:
            if tw in wds_list:
                for hyper in input[wds]:
                    out[wds].add(hyper)
    out2 = []
    for k, v in out.items():
        v = split_n_tag(v)
        k = k.replace("'","").replace('[', '').replace(']', '')
        k = [i.strip() for i in k.split(', ')]  # get a list of hyponyms
        k = split_n_tag(k)
        out2.append((k, v))
    print(out2[:1])
    
    return out2

if __name__ == "__main__":
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
        wds = wds.replace('"', '')
        par_wds = par_wds.replace('"', '')
        par_wds = par_wds.replace("'", '"')
        par_wds = json.loads(par_wds)
    
        temp[wds] = par_wds
    
    train = [i.strip() for i in open(args.train, 'r').readlines()]
    dev = [i.strip() for i in open(args.dev, 'r').readlines()]
    
    OUT_TRAIN = '%strains/' % OUT
    os.makedirs(OUT_TRAIN, exist_ok=True)
    
    out_train = convert(train, temp)
    print(len(out_train))
    print('===================================')
    out_dev = convert(dev, temp)
    print(len(out_dev))
    
    json.dump(out_train, open('%strain_%s_%s_%s.json.gz' % (OUT_TRAIN, POS, TEST, METHOD), 'w'))
    json.dump(out_dev, open('%sdev_%s_%s_%s.json.gz' % (OUT_TRAIN, POS, TEST, METHOD), 'w'))