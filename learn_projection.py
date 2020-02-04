#! python3
# coding: utf-8

from hyper_import_functions import learn_projection, load_embeddings, estimate_sims
from argparse import ArgumentParser
import pandas as pd
import sys, os
import numpy as np
import time
from configs import VECTORS, TAGS, MWE, EMB, OUT

vectors = VECTORS
tags = TAGS
mwe = MWE
emb_path = EMB
out = OUT

parser = ArgumentParser()
parser.add_argument('--trainfile', default='%s/%s_hypohyper_train.tsv.gz' % (out, vectors),
                    help="0.8 train of pre-processed training_data",
                    type=os.path.abspath)
parser.add_argument('--lmbd', action='store', type=float, default=0.0)
parser.add_argument('--skip_oov', action='store_true', help='Skip OOV entries?')

args = parser.parse_args()

start = time.time()

datafile = args.trainfile

data = pd.read_csv(datafile, sep='\t', header=0)

## the words are already lowercased and tagged
print(data.head(), file=sys.stderr)

hyponyms = data.hyponym.values
hypernyms = data.hypernym.values

print('Current embedding model:', emb_path.split('/')[-1], file=sys.stderr)
model = load_embeddings(emb_path)

print('Inferring vectors...', file=sys.stderr)

source_vecs = []
target_vecs = []

## in the train data, hyponym--hyperonym are not necessarily one2one correspondences
## multiple hypo-hypernymic relations are typical for polysemantic words,
# e.g CASE -> example/instance; CASE -> container; CASE -> a piece of furniture; CASE -> a set of facts
mult_hypernyms = {}  # Dictionary of hypernyms corresponding to each hyponym

for hyponym, hypernym in zip(hyponyms, hypernyms):
    if args.skip_oov:
        if hyponym not in model.vocab or hypernym not in model.vocab:
            continue
    
    if hyponym not in mult_hypernyms:
        mult_hypernyms[hyponym] = []
        
    if hyponym in model and hypernym in model: ## this is kinda taken care of earlier but okey, just double-checking
        mult_hypernyms[hyponym].append(hypernym)
        source_vec = model[hyponym]
        target_vec = model[hypernym]
        source_vecs.append(source_vec)
        target_vecs.append(target_vec)
    else:
        print(hyponym, hypernym, 'not found!', file=sys.stderr)

print('Whole train dataset shape:', len(source_vecs), file=sys.stderr)
print('Learning projection matrix...', file=sys.stderr)

transforms = learn_projection((source_vecs, target_vecs), model, lmbd=args.lmbd) ## this returns the transformation matrix
print('Transformation matrix created', transforms.shape, file=sys.stderr)

np.save('%s/%s_projection.npy' % (out, vectors), transforms)

end = time.time()
training_time = int(end - start)
print('Training data re-formated in %s minutes' % str(round(training_time/60)))