#! python3
# coding: utf-8

from smart_open import open
import json
import sys

predictionfile = sys.argv[1]
predictions = json.load(open(predictionfile, 'r'))

for word in predictions:
    for synset in predictions[word]:
        print('\t'.join([word, synset, 'whatever']))


