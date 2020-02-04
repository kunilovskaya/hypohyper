#! python3
# coding: utf-8

from hyper_imports import load_embeddings, predict
from argparse import ArgumentParser
import pandas as pd
import sys
from evaluate import get_score
import numpy as np
import time

from configs import VECTORS, EMB_PATH, OUT, POS

parser = ArgumentParser()
parser.add_argument('--testfile', default='%strains/%s_%s_test.tsv.gz' % (OUT, VECTORS, POS), help='0.2 of the training data reserved for intrinsic testing')
parser.add_argument('--projection', default='%sprojections/%s_%s_projection.npy' % (OUT, VECTORS, POS), help='.npy, the transformation matrix leanrt in the previous step')
parser.add_argument('--nr', type=int, default=10, help='Number of candidates.)

args = parser.parse_args()

start = time.time()

datafile = args.testfile

data = pd.read_csv(datafile, sep='\t', header=0)
print(data.head(), file=sys.stderr)

hyponyms = data.hyponym.values
hypernyms = data.hypernym.values

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

projection = np.load(args.projection)

ground_truth = {}  # Gold dictionary of hypernyms corresponding to each hyponym
predicted = {}  # Predicted dictionary of hypernyms corresponding to each hyponym
set_doubles = []

count_oov = 0
count_duplicate_hypo = 0 ## polisemy

for hyponym, hypernym in zip(hyponyms, hypernyms):
    if hypernym not in model.vocab or hyponym not in model.vocab:
        count_oov += 1
        continue
    if hyponym in ground_truth:
        set_doubles.append(hyponym)
        count_duplicate_hypo += 1
        
    if hyponym not in ground_truth:
        ground_truth[hyponym] = [] ## this is how i get the list of golden hypernyms
        
    ground_truth[hyponym].append(hypernym) ### we can end up with a list of hypernyms for one hypo
    
print('Cases of duplicate hyponyms in 0.2 test %d' % len(set(set_doubles)))
print('OOV: %d' % count_oov)
print('We will make predictions for %d unique hyponyms' % len(ground_truth), file=sys.stderr)

print('Making predictions...', file=sys.stderr)
counter = 0
hyper_collector = []
for hyponym in ground_truth:
    if hyponym in predicted:
        continue
        
    candidates, predicted_vector = predict(hyponym, model, projection, topn=args.nr)
    hyper_collector.append(predicted_vector)
    
    ## get a list of most the words most similar to the predicted hypernym vector
    candidates = [i[0] for i in candidates if i[0] != hyponym][:10]
    predicted[hyponym] = candidates

    if counter % 1000 == 0:
        print('%d hyponyms processed out of %d total' % (counter, len(ground_truth)),
              file=sys.stderr)
        # Want to see predictions in real time?
        print(hyponym, '\t', candidates)
        print('GROUND TRUTH: %s' % ground_truth[hyponym])
    counter += 1
## this is based on the intersection of lemmas, not synsets ids
mean_ap, mean_rr = get_score(ground_truth, predicted)
print("MAP: {0}\nMRR: {1}\n".format(mean_ap, mean_rr), file=sys.stderr)

end = time.time()
training_time = int(end - start)
print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('DONE evaluation step 3.\n Intrinsic Evaluation: %s minutes' % str(round(training_time/60)))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

