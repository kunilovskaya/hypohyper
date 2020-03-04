#! python3

# coding: utf-8

import sys

from hyper_imports import new_preprocess_mwe

for line in sys.stdin:
    res = line.strip().split('\t')
    (words, synsets) = res
    words = words.split(',')
    for word in words:
        lemma = new_preprocess_mwe(word.strip(), tags=True, pos='VERB')[0]
        print('\t'.join([lemma, synsets.strip()]))
