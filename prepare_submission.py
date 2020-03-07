#! python3
# coding: utf-8

from smart_open import open
import json
import sys
import zipfile
import os


predictionfile = sys.argv[1]
predictions = json.load(open(predictionfile, 'r'))

outfile = predictionfile.replace('.json', '.tsv')

with open(outfile, 'w') as f:
    for word in predictions:
        for synset in predictions[word]:
            f.write('\t'.join([word, synset, 'whatever']) + '\n')

zipped = zipfile.ZipFile(outfile.replace('.tsv', '.zip'), "w")
zipped.write(outfile)
os.remove(outfile)
