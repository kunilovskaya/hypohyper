#! python3
# coding: utf-8

from hyper_import_functions import load_embeddings, predict, get_vector
from argparse import ArgumentParser
import os, sys
from smart_open import open
import pickle
import numpy as np

from configs import VECTORS, TAGS, EMB, OUT, RANDOM_SEED, FT_EMB

tags = TAGS
emb_path = EMB

parser = ArgumentParser()
parser.add_argument('--provided_test', default='input/data/public_test/nouns_public.tsv', help='a list of hyponyms to attach to ruWordNet taxonomy', type=os.path.abspath)
parser.add_argument('--projection', default='%s%s_projection.npy' % (OUT, VECTORS), help='.npy, the transformation matrix leanrt in the previous step')
parser.add_argument('--oov_buster', default=FT_EMB, help='path to fasttext model to get vectors for OOV in provided test')
parser.add_argument('--nr', type=int, default=10, help='Number of candidates')

args = parser.parse_args()

datafile = args.provided_test

test_hyponyms = [i.strip().lower() for i in open(datafile, 'r').readlines()]

print('Current embedding model:', emb_path.split('/')[-1], file=sys.stderr)
model = load_embeddings(emb_path)

if tags == True:
    vec = get_vector('дом_NOUN', emb=model)
    print('%%%%%%%%%%%%%%%%%%%%%', len(vec))
if tags == False:
    vec = get_vector('дом', emb=model)
    print('=====================', len(vec))

projection = np.load(args.projection)

predicted = {}  # Predicted dictionary of hypernyms corresponding to each hyponym

print('We will make predictions for %d hyponyms' % len(test_hyponyms), file=sys.stderr)

print('Making predictions...', file=sys.stderr)
counter = 0
test_in_voc = []
oov_in_test = []
hyper_collector = []
for hyponym in test_hyponyms:
    # print(hyponym)
    try:
        candidates, predicted_vector = predict(hyponym, model, projection, topn=args.nr)
        test_in_voc.append(hyponym.upper()) ## formatting to meet ruWordNet conventions
        hyper_collector.append(predicted_vector)
    except:
        candidates = None
        oov_in_test.append(hyponym)

    if counter % 100 == 0:
        print('%d hyponyms processed out of %d total' % (counter, len(test_hyponyms )),
              file=sys.stderr)
        # Want to see predictions in real time?
        print(hyponym, '\t', candidates)
    counter += 1
print('Number of test words not in embeddings: %d (%d%%)' % (len(oov_in_test), len(oov_in_test)/len(test_hyponyms)*100))
print('OOV in test\n%s' % oov_in_test)
print('Saving the predicted vectors and the list of preprocessed %d test hyponyms that are found in vectors' % (len(test_hyponyms)-len(oov_in_test)))

np.save('%s%s_hyper_collector.npy' % (OUT, VECTORS), hyper_collector)
np.save('%s%s_preprocessed_test.npy' % (OUT, VECTORS), test_in_voc)
if len(hyper_collector) == len(test_in_voc):
    print('Sanity test: passed')
