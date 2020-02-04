### methods to identify the hypernym synset given a vector for a potential hypernym produced by hyponym -> hypernym projecting model
'''
for vector comparison in the measure_sims.py script
    (single) get embeddings only for single word senses
    (multi) consider single words and (if a sense is an MWE) consider main word's embedding
        (this should be limited to the synsets with no single word representations only?)
        this yields a single word duplicates from different synsets
        e.g. there are at least 10 synsets where ЗНАК is the head word of one of the senses lexicalized as a MWE (152660-N, 118639-N, 107519-N, 154560-N)
'''
import os
from hyper_imports import load_embeddings, get_vector, preprocess_mwe
from xml.dom import minidom
import numpy as np
import time

from configs import VECTORS, TAGS, EMB_PATH, OUT, RUWORDNET, POS, MODE

start = time.time()

########## LOAD resources ##########

model = load_embeddings(EMB_PATH)
emb_voc = model.vocab
if TAGS == True:
    vec = get_vector('дом_NOUN', emb=model)
    print('%%%%%%%%%%%%%%%%%%%%%', len(vec))
if TAGS == False:
    vec = get_vector('дом', emb=model)
    print('====================', len(vec))


####### parse ruwordnet and get a list of (synset_id, word) tuples for both one word and heads in MWE)
if POS == 'NOUN':
    senses = '%ssenses.N.xml' % RUWORDNET
elif POS == 'VERB':
    senses = '%ssenses.V.xml' % RUWORDNET
else:
    senses = None
    print('Not sure which PoS-domain you want from ruWordNet')

doc = minidom.parse(senses)
parsed_senses = doc.getElementsByTagName("sense")
all_id_senses = []
## how many synsets DON'T have single word members at all? -- 19.30%
print('Total number of senses %d' % len(parsed_senses))
count_main = 0
ids = []
for sense in parsed_senses:
    
    id = sense.getAttributeNode('synset_id').nodeValue
    name = sense.getAttributeNode("name").nodeValue
    main_wd = sense.getAttributeNode("main_word").nodeValue
    
    ids.append(id)
    if len(name.split()) == 0:
        item = None
        print('Missing name for a sense in synset %s' % id)
    
    if MODE == 'single':
        if len(name.split()) == 1:
            if TAGS == True:
                if POS == 'NOUN':
                    item = name.lower() + '_NOUN'
                elif POS == 'VERB':
                    item = name.lower() + '_VERB'
                else:
                    item = None
                    print('Which PoS part of WordNet are we dealing with?')
            else:
                item = name.lower()
                
            all_id_senses.append((id, item))
        else:
            continue
        
    ## this this supposed to include vectors for main components of MWE only if this synset has no single_word representation or if MWE is found in vectors
    elif MODE == 'main':
        if len(name.split()) == 1:
            if TAGS == True:
                if POS == 'NOUN':
                    item = name.lower() + '_NOUN'
                elif POS == 'VERB':
                    item = name.lower() + '_VERB'
                else:
                    item = None
                    print('Which PoS part of WordNet are we dealing with?')
            else:
                item = name.lower()

            all_id_senses.append((id, item))
            
        ## TODO: apply this condition only to synsets with no single_word representation;
        if len(name.split()) > 1:
            item = preprocess_mwe(name, tags=TAGS, pos=POS) ## this is compatible with embeddings already (lower, tagged)
            if item in emb_voc:
                ### adding the few cases of MWE in embeddings vocabulary
                all_id_senses.append((id, item))
            else:
                ## only if the respective synset has not been already added; no garantee that it has no single word lemmas further down
                count_main += 1
                if TAGS == True:
                    if POS == 'NOUN':
                        item = main_wd.lower() + '_NOUN'
                    elif POS == 'VERB':
                        item = main_wd.lower() + '_VERB'
                    else:
                        item = None
                        print('Which PoS part of WordNet are we dealing with?')
                else:
                    item = main_wd.lower()
                
                all_id_senses.append((id, item))
                    ## TODO deduplicate tuples
    else:
        print('What do you want to do with senses that are lexicalised as MWE?')

count_oov = 0

vectorized_senses_index = []
vectorized_senses = []

for id,item in all_id_senses:
    try:
        vect = get_vector(item, emb=model)
        if TAGS == True:
            item = item[:-5].upper()
        else:
            item = item.upper()
        vectorized_senses_index.append((id, item))
        vectorized_senses.append(vect)
    except:
        vect = None
        count_oov += 1
    
vectorized_sens_index = np.array(vectorized_senses_index)
vectorized_sens = np.array(vectorized_senses)

if len(vectorized_sens_index) == len(vectorized_sens):
    print('Sanity test: passed')

print('Total number of synsets: %d' % len(set(ids))) ##len(synset_seen_already)
print('Number of included senses (%s mode): %d' % (MODE, len(all_id_senses)))

if MODE == 'main':
    print('\tAdditional number of senses processed in the main mode: %d' % count_main)
print('Ratio of senses not found in embeddings to all senses_names considered in %s mode: %d%% (absolute number: %d)' % (MODE, (count_oov/len(all_id_senses)*100), count_oov))
## ## for mode=='single_wd': 16%; for mode=='main_wd': 8%

used_ids = set([i[0] for i in vectorized_sens_index])

print('I have %d vectorised senses from %d synsets in %s' % (len(vectorized_sens_index), len(used_ids), senses.split('/')[-1]))

outname = '%s_%s_%s_ruwordnet_vectorised.npz' % (MODE,VECTORS,POS)
OUT = '%sWordNet_vectorised/' % OUT
os.makedirs(OUT, exist_ok=True)

np.savez_compressed(OUT+outname, senses_index=vectorized_sens_index)

end = time.time()
training_time = int(end - start)
print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('DONE vectorising Wordnet step 5.\n Representing ruWordNet took %s minutes' % str(round(training_time/60)))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
