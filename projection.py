#! python3
# coding: utf-8

from hyper_import_functions import learn_projection, load_embeddings, estimate_sims
from argparse import ArgumentParser
import pandas as pd
import sys
import numpy as np


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--trainfile', required=True, action='store')
    parser.add_argument('--lmbd', action='store', type=float, default=0.0)
    parser.add_argument('--candidates', action='store', type=int, default=2)
    parser.add_argument('--threshold', action='store_true')
    parser.add_argument('--embedding', required=True)
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
    print('Tranformation matrix created', transforms.shape, file=sys.stderr)

    sim_average = None
    sim_std = None
    threshold = None
    if args.threshold:
        original_sims = []
        for source in mult_hypernyms:
            cur_sims = estimate_sims(source, mult_hypernyms[source], transforms, model)
            original_sims += cur_sims

        sim_average = np.average(original_sims)
        sim_std = np.std(original_sims)
        print('Average hypernym similarity to projection: %.3f' % sim_average, file=sys.stderr)
        print('Max hypernym similarity: %.3f' % np.max(original_sims), file=sys.stderr)
        print('Min hypernym similarity %.3f' % np.min(original_sims), file=sys.stderr)
        print('Standard deviation of hypernym similarities: %.3f' % sim_std, file=sys.stderr)
