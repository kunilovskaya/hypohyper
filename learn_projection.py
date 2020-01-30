#! python3
# coding: utf-8

from hyper_import_functions import learn_projection, load_embeddings, estimate_sims
from argparse import ArgumentParser
import pandas as pd
import sys, os
import numpy as np
import pickle
from smart_open import open
import time

# USAGE on u2:
# python3 code/hypohyper/learn_projection.py --trainfile proj/hypohyper/output/araneum_hypohyper_train.tsv.gz
# --emb_name araneum --emb_path resources/emb/araneum_upos_skipgram_300_2_2018.vec.gz --out proj/hypohyper/output/


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument('--trainfile', default='output/araneum_hypohyper_train.tsv.gz',
                        help="0.8 train of pre-processed training_data",
                        type=os.path.abspath)
    parser.add_argument('--lmbd', action='store', type=float, default=0.0)
    parser.add_argument('--threshold', action='store_true')
    parser.add_argument('--emb_name', default='araneum',
                        help="arbitrary name of the embedding for output formatting purposes: rdt, araneum, cc, other")
    parser.add_argument('--emb_path', default='input/resources/araneum_upos_skipgram_300_2_2018.vec.gz',
                        help="path to embeddings file")
    parser.add_argument('--out', default='output/', help="path to pickle with saved projection", type=os.path.abspath)
    
    args = parser.parse_args()

    start = time.time()
    
    datafile = args.trainfile
    modelfile = args.emb_path
    data = pd.read_csv(datafile, sep='\t', header=0)
    
    ## the words are already lowercased and tagged
    print(data.head(), file=sys.stderr)

    hyponyms = data.hyponym.values
    hypernyms = data.hypernym.values

    print('Current embedding model:', modelfile, file=sys.stderr)
    model = load_embeddings(modelfile)

    print('Inferring vectors...', file=sys.stderr)

    source_vecs = []
    target_vecs = []
    
    ## in the train data, hyponym--hyperonym are not necessarily one2one correspondences
    ## multiple hypo-hypernymic relations are typical for polysemantic words,
    # e.g CASE -> example/instance; CASE -> container; CASE -> a piece of furniture; CASE -> a set of facts
    mult_hypernyms = {}  # Dictionary of hypernyms corresponding to each hyponym

    for hyponym, hypernym in zip(hyponyms, hypernyms):
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
        
        ## we are tacling polysemy by ing the similarities for
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
        with open('%s/%s_projection.pickle.gz' % (args.out, args.emb_name), 'wb') as f:
            pickle.dump(output, f)

    end = time.time()
    training_time = int(end - start)
    print('Training data re-formated in %s minutes' % str(round(training_time/60)))