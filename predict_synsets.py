import argparse
import csv
import os
import sys
import time
import zipfile
from operator import itemgetter
import numpy as np
from evaluate import get_score
from smart_open import open
import pandas as pd
from collections import defaultdict
from itertools import repeat
import json


from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, TAGS, TOPN, TEST, METHOD
from hyper_imports import popular_generic_concepts, load_embeddings, parse_taxonymy,read_xml, id2name_dict, wd2id_dict
from hyper_imports import lemmas_based_hypers, mean_synset_based_hypers,id2wds_dict, synsets_vectorized, get_random_test
from hyper_imports import cooccurence_counts

parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
# for ultimate results to submit use private instead of public
if TEST == 'codalab':
    if POS == 'NOUN':
        parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
if TEST == 'static' or TEST == 'random' or TEST == 'provided':
        parser.add_argument('--test', default='%strains/%s_%s_%s_%s_WORDS.txt' % (OUT, VECTORS, POS, TEST, METHOD), type=os.path.abspath)
        
parser.add_argument('--hyper_vectors', default='%spredicted_hypers/%s_%s_%s_%s_hypers.npy' % (OUT, VECTORS, POS, TEST, METHOD),
                    help="predicted vectors")
parser.add_argument('--more_info', default='%scooc_hypernyms_public_nouns_news.tsv' % OUT,
                    help="predicted vectors")

args = parser.parse_args()

start = time.time()

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

if POS == 'NOUN':
    senses = '%ssenses.N.xml' % RUWORDNET
    synsets = '%ssynsets.N.xml' % RUWORDNET
elif POS == 'VERB':
    senses = '%ssenses.V.xml' % RUWORDNET
    synsets = '%ssynsets.V.xml' % RUWORDNET
else:
    senses = None
    synsets = None
    print('Not sure which PoS-domain you want from ruWordNet')
    
## get a list of tuples (134530-N, кунгур_NOUN)
synsets_dict = defaultdict(list)
sens_index = parse_taxonymy(senses, tags=TAGS, pos=POS, mode=MODE, emb_voc=model.vocab)

for i in sens_index:
    synset = i[0]
    ## <sense lemma="НЕСОБЛЮДЕНИЕ ПРАВИЛО ТОРГОВЛЯ" name="НЕСОБЛЮДЕНИЕ ПРАВИЛ ТОРГОВЛИ" ... /sense>
    name = i[1]
    synsets_dict[name].append(synset) ## a dict that reverses the list of tuples (134530-N, кунгур_NOUN) to get кунгур_NOUN:['134530-N']

parsed_syns = read_xml(synsets)
synsets_names = id2name_dict(parsed_syns)# a dict of format 144031-N:АУТИЗМ
synset_words = id2wds_dict(parsed_syns) # a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]
word_syn_ids = wd2id_dict(synset_words) # ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]

if OOV_STRATEGY == 'top-hyper':
    if POS == 'NOUN':
        rel_path = '%ssynset_relations.N.xml' % RUWORDNET
    elif POS == 'VERB':
        rel_path = '%ssynset_relations.V.xml' % RUWORDNET
    else:
        rel_path = None
        print('Which PoS?')
    top_ten = popular_generic_concepts(rel_path)
else:
    top_ten = None

test = [i.strip() for i in open(args.test, 'r').readlines()]

hyper_vecs = np.load(args.hyper_vectors, allow_pickle=True)

OUT = '%sresults/' % OUT
os.makedirs(OUT, exist_ok=True)
outfile = open('%s%s_%s_%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY, TEST, METHOD), 'w')
writer = csv.writer(outfile, dialect='unix', delimiter='\t', lineterminator='\n', escapechar='\\', quoting=csv.QUOTE_NONE)

counter = 0
nosamename = 0
pred_dict = defaultdict(list)
for hypo, hyper_vec in zip(test, hyper_vecs):
    if not np.any(hyper_vec):
        for line in top_ten: # synset ids already
            row = [hypo.strip(), line.strip(), 'dummy']
            writer.writerow(row)
            pred_dict[hypo.strip()].append(line.strip())
    else:
        if METHOD == 'base':
            # (default) get a list of (synset_id, hypernym_word, sim)
            this_hypo_res = lemmas_based_hypers(hypo, vec=hyper_vec, emb=model, topn=TOPN, dict_w2ids=synsets_dict) #кунгур_NOUN:['134530-N']
        
        elif METHOD == 'deworded':
            identifier_tuple, syn_vectors = synsets_vectorized(emb=model, worded_synsets=synset_words,
                                                               named_synsets=synsets_names, tags=TAGS, pos=POS)
            print('Number of vectorised synsets', len(syn_vectors))
            ## gets a list of (synset_id, hypernym_synset_name, sim); identifier_tuple is 134530-N, КУНГУР
            this_hypo_res = mean_synset_based_hypers(hypo, vec=hyper_vec, syn_ids=identifier_tuple, syn_vecs=syn_vectors, topn=10)
        
        elif 'corpus-informed' in METHOD:
            ## load the lists of hypernyms that coocur with the given hyponyms
            lines = open(args.more_info, 'r').readlines()
            freqs_dict = {}
            for line in lines:
                res = line.split('\t')
                hyponym, hypernyms = res
                freqs_dict[hyponym] = json.loads(hypernyms)
            
            this_hypo_res = cooccurence_counts(hypo, vec=hyper_vec, emb=model, topn=TOPN, dict_w2ids=synsets_dict, corpus_freqs=freqs_dict, method=METHOD) # кунгур_NOUN:['134530-N']
        else:
            this_hypo_res = None
            print('Any other methods to improve performance?')
            
        if counter % 500 == 0:
            print('%d hyponyms processed out of %d total' % (counter, len(test)),
                  file=sys.stderr)
            ## Want to see predictions in real time?
            print('%s: %s' % (hypo, this_hypo_res))
        
        counter += 1
        
        for line in this_hypo_res:
            row = [hypo.strip(), line[0].strip(), line[1].strip()]
            writer.writerow(row)
            pred_dict[hypo.strip()].append(line[0].strip())
outfile.close()

first3pairs_pred = {k: pred_dict[k] for k in list(pred_dict)[:3]}
print('PRED:', first3pairs_pred, file=sys.stderr)

if TEST == 'random' or TEST == 'static' or TEST == 'provided':
    json.dump(pred_dict, open('%s%s_%s_%s_pred.json' % (OUT, POS, TEST, METHOD), 'w'))
# print('Number of samename hypernyms excluded',nosamename)

elif TEST == 'codalab':
    # print('===Look at OOV===')
    # print('АНИСОВКА', [synsets_names[name] for name in pred_dict['АНИСОВКА']])
    # print(pred_dict['ВЭЙП'])
    # print(pred_dict['ДРЕСС-КОД'])
    # upload this archive to the site
    archive_name = '%s_%s_%s_%s_%s_%s.zip' % (VECTORS, POS, MODE, OOV_STRATEGY, TEST, METHOD)
    with zipfile.ZipFile(OUT + archive_name, 'w') as file:
        file.write('%s%s_%s_%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY, TEST, METHOD),
                   '%s_%s_%s_%s_%s_%s.tsv' % (VECTORS, POS, MODE, OOV_STRATEGY, TEST, METHOD))
    
end = time.time()
training_time = int(end - start)

print('=== DONE: %s has run ===\nMeasuring similarity and formatting output was done in %s minutes' % (os.path.basename(sys.argv[0]),str(round(training_time / 60))))
