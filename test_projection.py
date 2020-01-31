#! python3
# coding: utf-8

from hyper_import_functions import load_embeddings, predict
from argparse import ArgumentParser
import pandas as pd
import sys
from evaluate import get_score
import numpy as np
import time

from configs import VECTORS, TAGS, MWE, EMB, OUT, RUWORDNET, RANDOM_SEED

vectors = VECTORS
tags = TAGS
mwe = MWE
emb_path = EMB
out = OUT
ruWN = RUWORDNET
RANDOM_SEED = RANDOM_SEED

parser = ArgumentParser()
parser.add_argument('--testfile', default='%s%s_hypohyper_test.tsv.gz' % (out, vectors), help='0.2 of the training data reserved for intrinsic testing')
parser.add_argument('--projection', default='%s%s_projection.npy' % (out, vectors), help='.npy, the transformation matrix leanrt in the previous step')
parser.add_argument('--oov', action='store_true', help='if true, OOV targets are skipped')
parser.add_argument('--nr', type=int, default=10, help='Number of candidates')

args = parser.parse_args()

start = time.time()

datafile = args.testfile

data = pd.read_csv(datafile, sep='\t', header=0)
print(data.head(), file=sys.stderr)

hyponyms = data.hyponym.values
hypernyms = data.hypernym.values

print('Current embedding model:', emb_path.split('/')[-1], file=sys.stderr)
model = load_embeddings(emb_path)

projection = np.load(args.projection)

ground_truth = {}  # Gold dictionary of hypernyms corresponding to each hyponym
predicted = {}  # Predicted dictionary of hypernyms corresponding to each hyponym

count_oov = 0
count_duplicate_hypo = 0 ## polisemy

for hyponym, hypernym in zip(hyponyms, hypernyms):
    if args.oov:
        if hypernym not in model.vocab or hyponym not in model.vocab:
            count_oov += 1
            continue
    if hyponym in ground_truth:
        count_duplicate_hypo += 1
        
    if hyponym not in ground_truth:
        ground_truth[hyponym] = []
    ground_truth[hyponym].append(hypernym)
    
print('Duplicate hyponyms in test %d' % count_duplicate_hypo)
print('OOV: %d' % count_oov)
print('We will make predictions for %d hyponyms' % len(ground_truth), file=sys.stderr)

print('Making predictions...', file=sys.stderr)
counter = 0
hyper_collector = []
for hyponym in ground_truth:
    if hyponym in predicted:
        continue
    candidates, predicted_vector = predict(hyponym, model, projection, topn=args.nr)
    hyper_collector.append(predicted_vector)

    if counter % 1000 == 0:
        print('%d hyponyms processed out of %d total' % (counter, len(ground_truth)),
              file=sys.stderr)
        # Want to see predictions in real time?
        print(hyponym, '\t', candidates)
    counter += 1

mean_ap, mean_rr = get_score(ground_truth, predicted)
print("MAP: {0}\nMRR: {1}\n".format(mean_ap, mean_rr), file=sys.stderr)

end = time.time()
training_time = int(end - start)
print('Intrinsic Evaluation: %s minutes' % str(round(training_time/60)))

