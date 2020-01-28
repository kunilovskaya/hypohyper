#! python3
# coding: utf-8

from hyper_import_functions import learn_projection, load_embeddings, estimate_sims
from argparse import ArgumentParser
import pandas as pd
import sys
import numpy as np
import pickle
from smart_open import open


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--trainfile', required=True)
    parser.add_argument('--lmbd', action='store', type=float, default=0.0)
    parser.add_argument('--threshold', action='store_true')
    parser.add_argument('--embedding', required=True)
    parser.add_argument('--out', help="path to pickle with saved projection")
    args = parser.parse_args()

    datafile = args.trainfile
    modelfile = args.embedding
    data = pd.read_csv(datafile, sep='\t', header=0)
    print(data.head(), file=sys.stderr)

    hyponyms = data.hyponym.values
    hypernyms = data.hypernym.values

    print('Current embedding model:', modelfile, file=sys.stderr)
    model = load_embeddings(modelfile)

    print('Inferring vectors...', file=sys.stderr)

    source_vecs = []
    target_vecs = []

    mult_hypernyms = {}  # Dictionary of hypernyms corresponding to each hyponym

    for hyponym, hypernym in zip(hyponyms, hypernyms):
        if hyponym not in mult_hypernyms:
            mult_hypernyms[hyponym] = []

        if hyponym in model and hypernym in model:
            mult_hypernyms[hyponym].append(hypernym)
            source_vec = model[hyponym]
            target_vec = model[hypernym]
            source_vecs.append(source_vec)
            target_vecs.append(target_vec)
        else:
            print(hyponym, hypernym, 'not found!', file=sys.stderr)

    print('Whole train dataset shape:', len(source_vecs), file=sys.stderr)
    print('Learning projection matrix...', file=sys.stderr)
    transforms = learn_projection((source_vecs, target_vecs), model, lmbd=args.lmbd)
    print('Transformation matrix created', transforms.shape, file=sys.stderr)

    sim_average = None
    sim_std = None
    threshold = None
    if args.threshold:
        print('Finding the optimal threshold across %d hyponyms' % len(mult_hypernyms),
              file=sys.stderr)
        original_sims = []
        for source in mult_hypernyms:
            cur_sims = estimate_sims(source, mult_hypernyms[source], transforms, model)
            original_sims += cur_sims

        sim_average = np.average(original_sims)
        sim_std = np.std(original_sims)
        threshold = sim_average - sim_std
        print('Average hypernym similarity to projection: %.3f' % sim_average, file=sys.stderr)
        print('Max hypernym similarity: %.3f' % np.max(original_sims), file=sys.stderr)
        print('Min hypernym similarity %.3f' % np.min(original_sims), file=sys.stderr)
        print('Standard deviation of hypernym similarities: %.3f' % sim_std, file=sys.stderr)
        print('Resulting threshold: %.3f' % threshold, file=sys.stderr)

    if args.out:
        output = {'threshold': threshold, 'projection': transforms}
        with open('{filename}.pickle.gz'.format(filename=args.out), 'wb') as f:
            pickle.dump(output, f)

