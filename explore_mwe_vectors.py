import argparse
import csv
import os
import sys
import time
import zipfile
import numpy as np
from smart_open import open
from collections import defaultdict
import json

from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, FT_EMB, TAGS, vecTOPN, \
    TEST, METHOD, FILTER_1, FILTER_2
from hyper_imports import popular_generic_concepts, load_embeddings, filtered_dicts_mainwds_option,read_xml, id2name_dict

parser = argparse.ArgumentParser()
parser.add_argument('--tagged', default='lists/ruWordNet_names_pos.txt',
                    help="tagged 68K words and phrases from ruWordNet")
args = parser.parse_args()

words = set()

for line in open(args.tagged, 'r').readlines():
    line = line.strip()
    if line in ['ние_NOUN', 'к_NOUN', 'то_PRON', 'тот_DET', 'мочь_VERB', 'чать_VERB',
                'нать_VERB', 'аз_NOUN', 'в_NOUN', 'дание_NOUN']:
        continue
    if len(line.split()) == 1:
        continue
    else:
        line = line.replace(' ','::')
        words.add(line)

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
emb = load_embeddings(EMB_PATH)

counter = 0
for word in words:
    if word in emb.vocab:
        # print('часть_NOUN::тело_NOUN' in emb.vocab)
        counter += 1
        
print('Ratio of MWE in dedicated emb to all MWE in ruWordNet: %.2f%%' % (counter/len(words)*100))
print('Absolute nums: %d, %d' % (counter,len(words)))
