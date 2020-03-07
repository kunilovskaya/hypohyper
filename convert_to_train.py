#! python3

# coding: utf-8

import sys
from hyper_imports import map_mwe, preprocess_mwe
from configs import POS, TAGS
from hyper_imports import new_preprocess_mwe

source = open('lists/ruWordNet_%s_names.txt' % POS, 'r').readlines()
source_tagged = open('lists/ruWordNet_%s_same-names_pos.txt' % POS, 'r').readlines()
mwe_map = map_mwe(names=source, same_names=source_tagged, tags=TAGS, pos=POS)
not_mapped = 0
with open('output/trains/classifier_train_nouns.tsv', 'w') as outfile:
    header = '\t'.join(['words','synsets'])
    outfile.write(header + '\n')
    for line in sys.stdin:
        res = line.strip().split('\t')
        (words, synsets) = res
        words = words.split(',')
        for word in words:
            lemma, err = new_preprocess_mwe(word.strip(), tags=TAGS, pos=POS, map_mwe_names=mwe_map)
            not_mapped += err
            row = '\t'.join([lemma, synsets.strip()])
            outfile.write(row + '\n')
            print('\t'.join([lemma, synsets.strip()]))
            
print('Words missing from the map', not_mapped)
