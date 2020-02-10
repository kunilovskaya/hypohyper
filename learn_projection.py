#! python3
# coding: utf-8

from hyper_imports import learn_projection, load_embeddings, estimate_sims
from argparse import ArgumentParser
import pandas as pd
import sys, os
import numpy as np
import time
from hyper_imports import filter_dataset, read_xml, id2name_dict, id2wds_dict, synsets_vectorized
from configs import VECTORS, EMB_PATH, OUT, POS, SKIP_OOV, METHOD, TEST, RUWORDNET, TAGS
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument('--trainfile', default='%strains/%s_%s_%s_%s_train.tsv.gz' % (OUT, VECTORS, POS, METHOD, TEST),
                    help="0.8 static train of pre-processed training_data, all UPPER",
                    type=os.path.abspath)

parser.add_argument('--lmbd', action='store', type=float, default=0.0)

args = parser.parse_args()

start = time.time()

datafile = args.trainfile

data = pd.read_csv(datafile, sep='\t', header=0)

## the words are already lowercased and tagged
# print(data.head(), file=sys.stderr)

hyponyms = data.hyponym.values
hypernyms = data.hypernym.values

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)


source_vecs = []
target_vecs = []

if METHOD == 'deworded':
    if POS == 'NOUN':
        synsets = '%ssynsets.N.xml' % RUWORDNET
    elif POS == 'VERB':
        synsets = '%ssynsets.V.xml' % RUWORDNET
    else:
        synsets = None
        print('Not sure which PoS-domain you want from ruWordNet')
        
    parsed_syns = read_xml(synsets)
    synsets_names = id2name_dict(parsed_syns)# a dict of format 144031-N:АУТИЗМ
    synset_words = id2wds_dict(parsed_syns) # a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]
    ## (id,name)
    identifier_tuple, syn_vectors = synsets_vectorized(emb=model, worded_synsets=synset_words,
                                                       named_synsets=synsets_names, tags=TAGS, pos=POS)
    print('Number of vectorised synsets', len(syn_vectors))
    lookup = defaultdict()
    for (id,name), vect in zip(identifier_tuple, syn_vectors):
        lookup[id] = vect
    
    for hyponym, hypernym in zip(hyponyms, hypernyms):
        source_vec = lookup[hyponym]
        target_vec = lookup[hypernym]
        source_vecs.append(source_vec)
        target_vecs.append(target_vec)
    
else:
    ## in the train data, hyponym--hyperonym are not necessarily one2one correspondences
    ## multiple hypo-hypernymic relations are typical for polysemantic words,
    # e.g CASE -> example/instance; CASE -> container; CASE -> a piece of furniture; CASE -> a set of facts
    # mult_hypernyms = {}  # Dictionary of hypernyms corresponding to each hyponym which was used for threhold, but not anymore
    
    for hyponym, hypernym in zip(hyponyms, hypernyms):
        if SKIP_OOV == True:
            if hyponym in model.vocab or hypernym in model.vocab: # good for static-train
                source_vec = model[hyponym]
                target_vec = model[hypernym]
                source_vecs.append(source_vec)
                target_vecs.append(target_vec)
            else:
                continue
                
        elif SKIP_OOV == False:
            source_vec = model[hyponym]
            target_vec = model[hypernym]
            source_vecs.append(source_vec)
            target_vecs.append(target_vec)
        else:
            print(hyponym, hypernym, 'not found!', file=sys.stderr)
    
print('Whole train dataset shape:', len(source_vecs), file=sys.stderr)
# print('Learning projection matrix...', file=sys.stderr)

transforms = learn_projection((source_vecs, target_vecs), model, lmbd=args.lmbd) ## this returns the transformation matrix
print('Transformation matrix created', transforms.shape, file=sys.stderr)

OUT = '%sprojections/' % OUT
os.makedirs(OUT, exist_ok=True)

np.save('%s%s_%s_%s_%s_projection.npy' % (OUT, VECTORS, POS, METHOD, TEST), transforms)

end = time.time()
training_time = int(end - start)
print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%s has run\nProjections learnt in %s minutes' % (os.path.basename(sys.argv[0]), str(round(training_time/60))))
if TEST == 'provided':
    print('We are using 100\% of the training-data')
if TEST == 'intrinsic':
    print('We are using 0.8 of the training data with fewer monosemantic words which were saved for test')
if TEST == 'random':
    print('Using the standard train/test random split')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
