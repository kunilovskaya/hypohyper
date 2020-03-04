import argparse
import sys, os
from hyper_imports import preprocess_mwe, new_preprocess_mwe, map_mwe
from configs import VECTORS, OUT, POS, TAGS, METHOD, TEST
import json
from collections import defaultdict
from smart_open import open

input = open('/home/u2/git/hypohyper/input/data/all_data_nouns.tsv', 'r')
if VECTORS == 'mwe-pos-vectors':
    source = open('lists/ruWordNet_names.txt', 'r').readlines()
    source_tagged = open('lists/ruWordNet_same-names_pos.txt', 'r').readlines()
    mwe_map = map_mwe(names=source, same_names=source_tagged, tags=TAGS, pos=POS)
else:
    mwe_map = None
errs = 0
with open('oldtags_all_data_nouns.tsv', 'w') as outfile:
    for line in input:
        item, ids = line.strip().split('\t')
        # print(ids)
        if mwe_map:
            out, err = new_preprocess_mwe(item, tags=TAGS, pos=POS, map_mwe_names=mwe_map)
            errs += err
            row = '\t'.join([out,ids])
            # print(row)
            outfile.write(row + '\n')
        else:
            out = preprocess_mwe(item, tags=TAGS, pos=POS)
            row = '\t'.join([out, ids])
            outfile.write(row + '\n')