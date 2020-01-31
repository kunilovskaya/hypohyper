### methods to identify the hypernym synset given a vector for a potential hypernym produced by hyponym -> hypernym projecting model
'''
for vector comparison in the measure_sims.py script
    (single) get embeddings only for single word senses
    (multi) consider single words and (if a sense is an MWE) consider main word's embedding
        (this should be limited to the synsets with no single word representations only?)
        this yields a single word duplicates from different synsets
        e.g. there are at least 10 synsets where ЗНАК is the head word of one of the senses lexicalized as a MWE (152660-N, 118639-N, 107519-N, 154560-N)
'''

from hyper_import_functions import load_embeddings, get_vector, preprocess_mwe
import argparse
from xml.dom import minidom
import numpy as np


from configs import VECTORS, TAGS, MWE, EMB, OUT, RUWORDNET, RANDOM_SEED

vectors = VECTORS
tags = TAGS
mwe = MWE
emb_path = EMB
out = OUT
ruWN = RUWORDNET
RANDOM_SEED = RANDOM_SEED

parser = argparse.ArgumentParser('Get identifiers and vectors for senses in ruWordNet')
parser.add_argument('--mode', default='single_wd', type=str, help="if you want to include vectors for main_words in MWE, replace single_wd with main_wd")
parser.add_argument('--senses', default='%ssenses.N.xml' % ruWN, help="the file with 'main_word' and 'lemma' attibutes to sense tag")

args = parser.parse_args()

########## LOAD resources ##########

model = load_embeddings(emb_path)
emb_voc = model.vocab
if tags == True:
    vec = get_vector('дом_NOUN', emb=model)
    print('%%%%%%%%%%%%%%%%%%%%%', len(vec))
if tags == False:
    vec = get_vector('дом', emb=model)
    print('====================', len(vec))


####### parse ruwordnet and get a list of (synset_id, word) tuples for both one word and heads in MWE)
doc = minidom.parse(args.senses)
parsed_senses = doc.getElementsByTagName("sense")
all_id_senses = []
## how many synsets DON'T have single word members at all? -- 19.30%
print('Total number of senses %d' % len(parsed_senses))
synset_seen_already = set()
captured_synsets = set()
count_main = 0
for sense in parsed_senses:
    
    id = sense.getAttributeNode('synset_id').nodeValue
    name = sense.getAttributeNode("name").nodeValue
    main_wd = sense.getAttributeNode("main_word").nodeValue
    
    if len(name.split()) == 0:
        item = None
        print('Missing name for a sense in synset %s' % id)
    
    if args.mode == 'single_wd':
        if len(name.split()) == 1:
            item = name
            synset_seen_already.update(id)
            all_id_senses.append((id, item))
        
    ## this this supposed to include vectors for main components of MWE only if this synset has no single_word representation or if MWE is found in vectors
    elif args.mode == 'main_wd':
        if len(name.split()) == 1:
            item = name
            synset_seen_already.update(id)
            all_id_senses.append((id, item))
            
        ## TODO: apply this condition only to synsets with no single_word representation -- DONE
        if len(name.split()) > 1:
            item = preprocess_mwe(name, tags=True)
            if item in emb_voc:
                synset_seen_already.update(id)
                all_id_senses.append((id, item))
            else:
                if not id in synset_seen_already:
                    captured_synsets.update(id) ## or .add(a)??
                    count_main += 1
                    item = main_wd
                    all_id_senses.append((id, item))
    else:
        print('What do you want to do with senses that are lexicalised as MWE?')
        
    
count_oov = 0

vectorized_senses_index = []
vectorized_senses = []

for id,item in all_id_senses:
    try:
        vect = get_vector(item.lower()+'_NOUN', emb=model)
        vectorized_senses_index.append((id, item))
        vectorized_senses.append(vect)
    except:
        vect = None
        count_oov += 1
        
vectorized_sens_index = np.array(vectorized_senses_index)
vectorized_sens = np.array(vectorized_senses)

if len(vectorized_sens_index) == len(vectorized_sens):
    print('Sanity test: passed')

if args.mode == 'main_wd':
    print('Number of cases captured: %d' % count_main)
    print('Number of no-single-word synsets salvaged via considering the main word of their MWE: %d' % len(
        captured_synsets)) ## TODO this returns suspiciously few cases (12)
print('Number of included senses (%s mode): %d' % (args.mode, len(all_id_senses)))
print('Ratio of senses not found in embeddings to all included in %s mode: %d%%' % (args.mode, (count_oov/len(all_id_senses)*100))) ## ## for mode=='single_wd': 16%; for mode=='main_wd': 8%
print('Absolute number of OOV senses', count_oov)

if args.mode == 'single_wd':
    outname = vectors+'_single_ruwordnet_vectorized.npz'
elif args.mode == 'main_wd':
    outname = vectors+'_main_ruwordnet_vectorized.npz'
else:
    print('What do you want to do with sense lexicalised as MWE')
    outname = None

np.savez_compressed(out+outname, senses_index=vectorized_sens_index,
                    senses_vectors=vectorized_sens)
