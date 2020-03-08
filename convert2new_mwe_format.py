import argparse
import sys, os
from hyper_imports import preprocess_mwe, new_preprocess_mwe, map_mwe
from configs import VECTORS, OUT, POS, TAGS, METHOD, TEST
import json
from collections import defaultdict
from smart_open import open

if POS == 'NOUNS':
    input = open('input/data/all_data_nouns.tsv', 'r')
elif POS == 'VERB': # АРЕНДОВАТЬ ЖИЛУЮ ПЛОЩАДЬ
    input = open('input/data/all_data_verbs.tsv', 'r')
    
if 'mwe' in VECTORS:
    
    source = open('lists/ruWordNet_%s_names.txt' % POS, 'r').readlines()
    source_tagged = open('lists/ruWordNet_%s_same-names_pos.txt' % POS, 'r').readlines()
    mwe_map = map_mwe(names=source, same_names=source_tagged, tags=TAGS, pos=POS)

    errs = 0
    with open('input/data/newtags_all_data_%s.tsv' % POS, 'w') as outfile:
        header = '\t'.join(['word', 'synsets'])
        outfile.write(header + '\n')
        for line in input:
            item, ids = line.strip().split('\t')
            out, err = new_preprocess_mwe(item, tags=TAGS, pos=POS, map_mwe_names=mwe_map)
            errs += err
            row = '\t'.join([out, ids])
            # print(row)
            outfile.write(row + '\n')
else:
    with open('input/data/oldtags_all_data_%s.tsv' % POS, 'w') as outfile:
        header = '\t'.join(['words', 'synsets'])
        outfile.write(header + '\n')
        for line in input:
            item, ids = line.strip().split('\t')
            out = preprocess_mwe(item, tags=TAGS, pos=POS)
            row = '\t'.join([out, ids])
            outfile.write(row + '\n')