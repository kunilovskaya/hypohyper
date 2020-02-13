from hyper_imports import load_embeddings, predict
from argparse import ArgumentParser
import os, sys
from smart_open import open
import numpy as np
import time

from configs import VECTORS, TAGS, EMB_PATH, OUT, FT_EMB, OOV_STRATEGY, POS, METHOD, TEST

parser = ArgumentParser()
# for ultimate results use private instead of public
if TEST == 'codalab':
    if POS == 'NOUN':
        parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
if TEST == 'provided':
        parser.add_argument('--test', default='%strains/%s_%s_%s_%s_WORDS.txt' % (OUT, VECTORS, POS, TEST, METHOD), type=os.path.abspath)
parser.add_argument('--projection', default='%sprojections/%s_%s_%s_%s_projection.npy' % (OUT, VECTORS, POS, TEST, METHOD))
parser.add_argument('--nr', type=int, default=10, help='Number of candidates')

args = parser.parse_args()

start = time.time()

## preprocess words
if TAGS == True:
    if POS == "NOUN":
        test_hyponyms = [i.strip().lower()+'_NOUN' for i in open(args.test, 'r').readlines()]
    elif POS == "VERB":
        test_hyponyms = [i.strip().lower()+'_VERB' for i in open(args.test, 'r').readlines()]
    else:
        test_hyponyms = None
else:
    test_hyponyms = [i.strip().lower() for i in open(args.test, 'r').readlines()]

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

## use FT to bust OOV in the input
if OOV_STRATEGY == 'ft-vector':
    print('FT embedding model:', FT_EMB.split('/')[-1], file=sys.stderr)
    ft_model = load_embeddings(FT_EMB)
else:
    ft_model = None

    
## TODO other strategies to eliminate OOV: use cooc stats
#### get a vector of the name of most frequent hypernym synset in the training data

projection = np.load(args.projection)

predicted = {}  # Predicted dictionary of hypernyms corresponding to each hyponym

print('We will make predictions for %d hyponyms' % len(test_hyponyms), file=sys.stderr)

counter = 0
test_preprocessed = []
oov_in_test = []
hyper_collector = []

for hyponym in test_hyponyms:
    if hyponym in model.vocab:
        candidates, predicted_vector = predict(hyponym, model, projection, topn=args.nr)
    else:
        if OOV_STRATEGY == 'ft-vector':
            if TAGS == True:
                candidates, predicted_vector = predict(hyponym[:-5], ft_model, projection, topn=args.nr)
                oov_in_test.append(hyponym)
            else:
                candidates, predicted_vector = predict(hyponym, ft_model, projection, topn=args.nr)
                oov_in_test.append(hyponym)
        elif OOV_STRATEGY == 'top-hyper':
            candidates = []
            oov_in_test.append(hyponym)
            predicted_vector = None #[0]  # placeholder for the absent vector
        else:
            candidates = []
            predicted_vector = None
            print('I am not sure how you are going to treat OOV in testset?')

    hyper_collector.append(predicted_vector)
    
    if counter % 500 == 0:
        print('%d hyponyms processed out of %d total' % (counter, len(test_hyponyms )),
              file=sys.stderr)
        # Want to see predictions in real time?
        print(hyponym, '\t', candidates)
    counter += 1

print('\n====== Number of OOV in test: %d (%d%%)' % (len(oov_in_test), len(oov_in_test)/len(test_hyponyms)*100))
# print('OOV in test\n%s\n' % oov_in_test)

OUT = '%spredicted_hypers/' % OUT
os.makedirs(OUT, exist_ok=True)

np.save('%s%s_%s_%s_%s_hypers.npy' % (OUT, VECTORS, POS, TEST, METHOD), hyper_collector)

end = time.time()
training_time = int(end - start)

print('=== %s has run. ===\nLearning hypernym vectors for the real testset: %s minutes' % (os.path.basename(sys.argv[0]), str(round(training_time / 60))))

