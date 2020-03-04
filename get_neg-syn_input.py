##  json вот с таким содержанием [('['усмешка_NOUN', 'ухмылка_NOUN'], ['преподнесение_NOUN', 'представление_NOUN'])]

import argparse
import sys, os
from hyper_imports import preprocess_mwe, new_preprocess_mwe, map_mwe
from configs import VECTORS, OUT, POS, TAGS, METHOD, TEST
import json
from collections import defaultdict
from smart_open import open

def split_n_tag(bad_list, mapping=None): ## list in, list out
    errs = 0
    res_list = []
    for i in bad_list:
        if ', ' not in i:
            if mapping != None:
                new_i, err = new_preprocess_mwe(i, tags=TAGS, pos=POS, map_mwe_names=mapping)
                res_list.append(new_i)
                errs += err
                continue
            else:
                i = preprocess_mwe(i, tags=TAGS, pos=POS)
                res_list.append(i)
        elif ', ' in i:
            items = [x.strip() for x in i.split(', ')]  # get a list of items in the string
            for wd in items:
                if mapping:
                    out, err = new_preprocess_mwe(wd, tags=TAGS, pos=POS, map_mwe_names=mapping)
                    errs += err
                    res_list.append(out)
                else:
                    ii = preprocess_mwe(ii, tags=TAGS, pos=POS)
                    res_list.append(ii)
    
    return res_list, errs

def convert(data, input, mapping=None):
    # data_hyper_ids = defaultdict(list)
    mapping_err = 0
    twords = []
    for i in data:
        res = i.strip().split('\t')
        tword, hyper_ids = res
        twords.append(tword.strip())
        
    out = defaultdict(set)
    
    
    for wds in input:  # can be a string of several hyponyms from source; vals are parents wordlists
        wds_list = [i.strip() for i in wds.split(', ')]  # get a list of hyponyms
        # if 'КОРРЕСПОНДЕНТСКИЙ ПУНКТ' in wds_list:
        #     print('GOTCHA')
        for tw in twords:
            if tw in wds_list:
                for hyper in input[wds]:
                    out[wds].add(hyper.strip())
        
    out2 = []
    for k0, v0 in out.items():
        v, err = split_n_tag(list(v0), mapping=mapping) # is a list of hypernyms
        mapping_err += err
        k = k0.replace("'","") #.replace('[', '').replace(']', '')
        k1 = [i.strip() for i in k0.split(', ')]  # get a list of hyponyms again
        k, err = split_n_tag(k1, mapping=mapping)
        mapping_err += err
        out2.append((k, v)) # tuples of lists
    print(out2[:2])
    print('Mapping complete with %d errors' % mapping_err)
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
    if VECTORS == 'mwe-pos-vectors':
        WN_names = open('lists/ruWordNet_%s_names.txt' % POS, 'r').readlines()
        WN_names_tagged = open('lists/ruWordNet_%s_same-names_pos.txt' % POS, 'r').readlines()
        mapped = map_mwe(names=WN_names, same_names=WN_names_tagged, tags=TAGS, pos=POS)
    else:
        mapped=None
        
    temp = {}
    for i in source:
        res = i.split('\t')
        _, wds, par_ids, par_wds = res
        ## get gid of double quotes inside strings: ПРЕМИЯ "ОСКАР"
        wds = wds.replace('"', '')
        par_wds = par_wds.replace('"', '')
        par_wds = par_wds.replace("'", '"')
        par_wds = json.loads(par_wds)
        # print(wds)
        temp[wds] = par_wds
    
    train = [i.strip() for i in open(args.train, 'r').readlines()]
    dev = [i.strip() for i in open(args.dev, 'r').readlines()]
    
    OUT_TRAIN = '%strains/' % OUT
    os.makedirs(OUT_TRAIN, exist_ok=True)
    
    out_train = convert(train, temp, mapping=mapped)
    
    print(len(out_train))
    print('===================================')
    out_dev = convert(dev, temp, mapping=mapped)
    print(len(out_dev))
    
    json.dump(out_train, open('%snew-tags_train_%s_%s_%s.json.gz' % (OUT_TRAIN, POS, TEST, METHOD), 'w'))
    json.dump(out_dev, open('%snew-tags_dev_%s_%s_%s.json.gz' % (OUT_TRAIN, POS, TEST, METHOD), 'w'))