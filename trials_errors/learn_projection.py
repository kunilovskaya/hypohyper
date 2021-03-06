from trials_errors.hyper_imports import learn_projection, load_embeddings
from argparse import ArgumentParser
import sys, os
import numpy as np
import time
from hyper_imports import preprocess_hypo, preprocess_wordpair, read_xml, id2name_dict, synsets_vectorized
from hyper_imports import filtered_dicts_mainwds_option
from configs import VECTORS, EMB_PATH, OUT, POS, SKIP_OOV, METHOD, TEST, RUWORDNET, TAGS, MWE, MODE
from collections import defaultdict
import json

parser = ArgumentParser()
parser.add_argument('--train', default='%strains/%s_%s_%s_%s_train.json' % (OUT, VECTORS, POS, TEST, METHOD),
                    help="unfiltered, (CAPS,CAPS) or (CAPS,id) pairs",
                    type=os.path.abspath)

parser.add_argument('--lmbd', action='store', type=float, default=0.0)

args = parser.parse_args()

start = time.time()

## filter the training data (ex. 'КИВИ', 'ОБЪЕКТ ЖИВОТНЫЙ МИР'), taking into account TAGS and POS and model.vocab
pairs = json.load(open(args.train, 'r')) # list of tuples

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

source_vecs = []
target_vecs = []

if METHOD == 'deworded': # or METHOD != 'deworded1'
    data = preprocess_hypo(pairs, tags=TAGS, mwe=MWE, pos=POS)  # list of tuples
    ## this is formated to lookup in embeddings
    hyponyms = [pair[0] for pair in data]
    hypernyms = [pair[1] for pair in data]
    
    oov_hypo = []
    oov_synsets = []
    synsets = None
    senses = None
    if POS == 'NOUN':
        senses = '%ssenses.N.xml' % RUWORDNET
        synsets = '%ssynsets.N.xml' % RUWORDNET
    elif POS == 'VERB':
        senses = '%ssenses.V.xml' % RUWORDNET
        synsets = '%ssynsets.V.xml' % RUWORDNET

    # this is where single/main word MODE is applied
    ## get {'родитель_NOUN': ['147272-N', '136129-N', '5099-N', '2655-N'], 'злоупотребление_NOUN': ['7331-N', '117268-N', '143946-N', '7332-N', '117269-N']}
    ## and its reverse
    _, id2lemmas = filtered_dicts_mainwds_option(senses, tags=TAGS, pos=POS, mode=MODE, emb_voc=model.vocab)
    
    parsed_syns = read_xml(synsets)
    synsets_names = id2name_dict(parsed_syns) # a dict of format 144031-N:АУТИЗМ
    # synset_words = id2wds_dict(parsed_syns) # a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]

    identifier_tuple, syn_vectors = synsets_vectorized(emb=model, id2lemmas=id2lemmas,
                                                       named_synsets=synsets_names)
    print('Number of vectorised synsets', len(syn_vectors))
    lookup = defaultdict()
    for (id,name), vect in zip(identifier_tuple, syn_vectors):
        lookup[id] = vect
    
    for hyponym, hypernym in zip(hyponyms, hypernyms):
        if hyponym not in model.vocab:
            oov_hypo.append(hyponym)
        if hypernym not in lookup.keys():
            oov_synsets.append(hypernym)
        if hyponym in model.vocab and hypernym in lookup.keys(): # this is where we filter thru emb.vocab and skip synsets with no singleword representation
            source_vec = model[hyponym]
            target_vec = lookup[hypernym]
            source_vecs.append(source_vec)
            target_vecs.append(target_vec)
        else:
            source_vec = None

    print('OOV unique hyponyms: ', len(set(oov_hypo)))
    print('OOV unique synsets: ', len(set(oov_synsets)))
    
else:
    data = preprocess_wordpair(pairs, tags=TAGS, mwe=MWE, pos=POS)  # list of tuples
    
    hyponyms = [pair[0] for pair in data]
    hypernyms = [pair[1] for pair in data]
    
    for hyponym, hypernym in zip(hyponyms, hypernyms):
        if SKIP_OOV == True:
            if hyponym in model.vocab and hypernym in model.vocab: # this is where filtering thru emb happens
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
transforms = learn_projection((source_vecs, target_vecs), model, lmbd=args.lmbd) ## this returns the transformation matrix
# print('Transformation matrix created', transforms.shape, file=sys.stderr)

OUT = '%sprojections/' % OUT
os.makedirs(OUT, exist_ok=True)

np.save('%s%s_%s_%s_%s_projection.npy' % (OUT, VECTORS, POS, TEST, METHOD), transforms)

end = time.time()
training_time = int(end - start)

print('METHOD == %s' % METHOD, file=sys.stderr)
print('TEST == %s' % TEST, file=sys.stderr)
print('Train is filtered with an embedding model', file=sys.stderr)
print('Training on %s pairs' % len(source_vecs), file=sys.stderr)
print('TRAIN: ', data[:3], file=sys.stderr)
print('=== %s has run ===\nProjections learnt in %s minutes\n' % (os.path.basename(sys.argv[0]), str(round(training_time/60))))

