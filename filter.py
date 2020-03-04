#! python3
# coding: utf-8

from smart_open import open
import json
import sys

testfile = sys.argv[1]
gold_words = json.load(open(testfile, 'r'))

testwords = set([w.lower()+'_NOUN' for w in gold_words.keys()])

for line in sys.stdin:
    res = line.strip().split('\t')
    (word, synsets) = res
    if word.strip() in testwords:
        continue
    print(line.strip())



