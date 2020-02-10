import argparse
import csv
import os
import pandas as pd
from smart_open import open
import json
from collections import defaultdict


## I need HYPO ['synset_id', 'synset_id'] format from
# (1) training data format test file in intrinsic mode
# (2) hypo_NOUN hyper_NOUN random split

from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, TAGS, TOPN, OPT, TEST
from hyper_imports import read_xml, wd2id_dict
from hyper_imports import id2wds_dict, get_random_test, get_intrinsic_test

parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
# for ultimate results to submit use private instead of public

if TEST == 'random':
    parser.add_argument('--truth', default='%strains/%s_%s_%s_test.tsv.gz' % (OUT, VECTORS, POS, TEST)) ## replace values(words with lists of associated ids (from wd2ids))
if TEST == 'intrinsic':
    parser.add_argument('--truth', default='%s/trains/static/%s_static-test.tsv' % (OUT, POS)) # use PARENTS col directly as a list of gold synset_ids for test items

args = parser.parse_args()

OUT = '%sgold_dicts/' % OUT
os.makedirs(OUT, exist_ok=True)

if POS == 'NOUN':
    synsets = '%ssynsets.N.xml' % RUWORDNET
elif POS == 'VERB':
    synsets = '%ssynsets.V.xml' % RUWORDNET
else:
    synsets = None
    print('Not sure which PoS-domain you want from ruWordNet')

parsed_syns = read_xml(synsets)
synset_words = id2wds_dict(parsed_syns) # a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]
## ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]
word_syn_ids = wd2id_dict(synset_words)


if TEST == 'random':
    # GOLD {'СМЕНА': [['118449-N', '106802-N']], 'СЕЙСМИЧНОСТЬ': [['106515-N', '106773-N'], ['106515-N']]}
    # PRED
    gold_dict = get_random_test(goldpath=args.truth, w2ids_d=word_syn_ids)
    
elif TEST == 'intrinsic':
    gold_dict = get_intrinsic_test(goldpath=args.truth)
    print('Test set for intrinsic evaluation is ready')
    
else:
    print('I am not sure which testset to use...')
    gold_dict = None
    
json.dump(gold_dict, open('%s%s_%s_gold.json' % (OUT, POS, TEST), 'w'))
