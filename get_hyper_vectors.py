#! python3
# coding: utf-8

from hyper_imports import load_embeddings, predict, get_vector
from argparse import ArgumentParser
import os, sys
from smart_open import open
import numpy as np
import time

from configs import VECTORS, TAGS, EMB_PATH, OUT, FT_EMB, OOV_STRATEGY, POS

parser = ArgumentParser()
if POS == 'NOUN':
    parser.add_argument('--provided_test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
if POS == 'VERB':
    parser.add_argument('--provided_test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
parser.add_argument('--projection', default='%sprojections/%s_%s_projection.npy' % (OUT, VECTORS, POS), help='.npy, the transformation matrix leanrt in the previous step')
parser.add_argument('--nr', type=int, default=10, help='Number of candidates')

args = parser.parse_args()

start = time.time()

datafile = args.provided_test

if TAGS == True:
    if POS == "NOUN":
        test_hyponyms = [i.strip().lower()+'_NOUN' for i in open(datafile, 'r').readlines()]
    elif POS == "VERB":
        test_hyponyms = [i.strip().lower()+'_VERB' for i in open(datafile, 'r').readlines()]
    else:
        test_hyponyms = None
else:
    test_hyponyms = [i.strip().lower() for i in open(datafile, 'r').readlines()]

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

#### use FT to bust OOV in the input
if OOV_STRATEGY == 'ft_vector':
    print('FT embedding model:', FT_EMB.split('/')[-1], file=sys.stderr)
    ft_model = load_embeddings(FT_EMB)

    
## TODO other strategies to eliminate OOV
#### get a vector of the name of most frequent hypernym synset in the training data

projection = np.load(args.projection)

predicted = {}  # Predicted dictionary of hypernyms corresponding to each hyponym

print('We will make predictions for %d hyponyms' % len(test_hyponyms), file=sys.stderr)

print('Making predictions...', file=sys.stderr)
counter = 0
test_preprocessed = []
oov_in_test = []
hyper_collector = []

for hyponym in test_hyponyms:
    # print(hyponym)
    if hyponym in model.vocab:
        candidates, predicted_vector = predict(hyponym, model, projection, topn=args.nr)
    else:
        if OOV_STRATEGY == 'ft_vector':
            if TAGS == True:
                candidates, predicted_vector = predict(hyponym[:-5], ft_model, projection, topn=args.nr)
                oov_in_test.append(hyponym)
            else:
                candidates, predicted_vector = predict(hyponym, ft_model, projection, topn=args.nr)
                oov_in_test.append(hyponym)
        elif OOV_STRATEGY == 'top_hyper':
            candidates = []
            oov_in_test.append(hyponym)
            predicted_vector = [0]  # placeholder for the absent vector

    hyper_collector.append(predicted_vector)
    
    if counter % 100 == 0:
        print('%d hyponyms processed out of %d total' % (counter, len(test_hyponyms )),
              file=sys.stderr)
        # Want to see predictions in real time?
        print(hyponym, '\t', candidates)
    counter += 1

print('\nOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
print('====== Number of OOV in test: %d (%d%%)' % (len(oov_in_test), len(oov_in_test)/len(test_hyponyms)*100))
print('OOV in test\n%s' % oov_in_test)
print('\nSaving the predicted vectors and the list of preprocessed %d test hyponyms that are found in vectors' % (len(test_hyponyms)-len(oov_in_test)))
print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\n')

OUT = '%spredicted_hypers/' % OUT
os.makedirs(OUT, exist_ok=True)

np.save('%s%s_%s_hyper_collector.npy' % (OUT, VECTORS, POS), hyper_collector)

if len(hyper_collector) == len(test_hyponyms):
    print('Sanity test: passed')
    print('Length of output equals the number of test words: %s' % len(hyper_collector))


end = time.time()
training_time = int(end - start)
print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('GOT hypernym vectors step 4.\n Learning hypernym vectors for the real testset: %s minutes' % str(round(training_time / 60)))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
