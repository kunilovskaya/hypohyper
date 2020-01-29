#! python3
# coding: utf-8

from hyper_import_functions import load_embeddings, predict
from argparse import ArgumentParser
import pandas as pd
import sys
from smart_open import open
import pickle
from evaluate import get_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--testfile', required=True)
    parser.add_argument('--projection', required=True)
    parser.add_argument('--embedding', required=True)
    parser.add_argument('--oov', action='store_true', help='If true, OOV targets are skipped')
    parser.add_argument('--nr', type=int, default=10, help='Number of candidates')
    parser.add_argument('--restrict',
                        help='Path to the file containing a list of allowed hypernyms')
    args = parser.parse_args()

    datafile = args.testfile
    modelfile = args.embedding
    data = pd.read_csv(datafile, sep='\t', header=0)
    print(data.head(), file=sys.stderr)

    hyponyms = data.hyponym.values
    hypernyms = data.hypernym.values
    allowed = set()

    print('Current embedding model:', modelfile, file=sys.stderr)
    model = load_embeddings(modelfile)


    if args.restrict:
        with open(args.restrict, 'r') as f:
            for line in f:
                lemma = line.strip()
                allowed.add(lemma)
        allowed = allowed & set(model.index2word)
        print('Using %d lemmas as possible candidates' % len(allowed), file=sys.stderr)

    pickle_file = open(args.projection, 'rb')
    pickle_data = pickle.load(pickle_file)

    threshold = pickle_data['threshold']
    print('Using threshold:', threshold, file=sys.stderr)
    projection = pickle_data['projection']

    ground_truth = {}  # Gold dictionary of hypernyms corresponding to each hyponym
    predicted = {}  # Predicted dictionary of hypernyms corresponding to each hyponym

    for hyponym, hypernym in zip(hyponyms, hypernyms):
        if args.oov:
            if hypernym not in model.vocab:
                continue
        if hyponym not in ground_truth:
            ground_truth[hyponym] = []
        ground_truth[hyponym].append(hypernym)

    print('We will make predictions for %d hyponyms' % len(ground_truth), file=sys.stderr)

    print('Making predictions...', file=sys.stderr)
    counter = 0
    for hyponym in ground_truth:
        if hyponym in predicted:
            continue
        candidates, predicted_vector = predict(hyponym, model, projection, topn=args.nr)

        if args.restrict:
            candidates = [w for w in candidates if w[0] in allowed]

        if threshold:
            # Filtering stage
            # We allow only candidates which are not further from the projection
            # than one sigma from the average similarity in the true set
            rejected = [c for c in candidates if c[1] < threshold]
            candidates = [c for c in candidates if c[1] >= threshold]
        else:
            rejected = []
        # End filtering stage

        candidates = [i[0] for i in candidates if i[0] != hyponym][:10]
        predicted[hyponym] = candidates

        if counter % 1000 == 0:
            print('%d hyponyms processed out of %d total' % (counter, len(ground_truth)),
                  file=sys.stderr)
            # Want to see predictions in real time?
            print(hyponym, '\t', candidates)
        counter += 1

    mean_ap, mean_rr = get_score(ground_truth, predicted)
    print("MAP: {0}\nMRR: {1}\n".format(mean_ap, mean_rr), file=sys.stderr)
