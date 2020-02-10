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

from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, TAGS, TOPN, OPT, TEST
from hyper_imports import popular_generic_concepts, load_embeddings, parse_taxonymy,read_xml, id2name_dict, wd2id_dict
from hyper_imports import lemmas_based_hypers, mean_synset_based_hypers,id2wds_dict, synsets_vectorized, get_random_test
from hyper_imports import get_intrinsic_test

parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
parser.add_argument('--effort', default='base', required=True) #'averaged_syns' #, 'corpus-informed' base
# for ultimate results to submit use private instead of public
if TEST == 'provided':
    if POS == 'NOUN':
        parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
if TEST == 'intrinsic' or TEST == 'random':
    if POS == 'NOUN':
        parser.add_argument('--test', default='%strains/%s_%s_%s_test4testing.txt' % (OUT, VECTORS, POS, TEST), type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='%strains/%s_%s_%s_test4testing.txt' % (OUT, VECTORS, POS, TEST), type=os.path.abspath)
        
parser.add_argument('--hyper_vectors', default='%spredicted_hypers/%s_%s_%s_%s_hypers.npy' % (OUT, VECTORS, POS, OPT, TEST),
                    help="predicted vectors")
parser.add_argument('--more_info', default='%scooc_hypernyms_public_nouns_news.tsv' % (OUT),
                    help="predicted vectors")

args = parser.parse_args()

start = time.time()

effort = args.effort

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
    
print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

sens_index = parse_taxonymy(senses, tags=TAGS, pos=POS, mode=MODE, emb_voc=model.vocab)

synsets_dict = defaultdict(list) ## defaultdict(set)

for i in sens_index:
    synset = i[0]
    ## <sense lemma="НЕСОБЛЮДЕНИЕ ПРАВИЛО ТОРГОВЛЯ" name="НЕСОБЛЮДЕНИЕ ПРАВИЛ ТОРГОВЛИ" ... /sense>
    name = i[1]
    synsets_dict[name].append(synset)
    # synsets_dict[name].add(synset)

parsed_syns = read_xml(synsets)
synsets_names = id2name_dict(parsed_syns)# a dict of format 144031-N:АУТИЗМ
synset_words = id2wds_dict(parsed_syns) # a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]
## ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]
word_syn_ids = wd2id_dict(synset_words)

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
outfile = open('%s%s_%s_%s_%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST, effort), 'w')
writer = csv.writer(outfile, dialect='unix', delimiter='\t', lineterminator='\n', escapechar='\\', quoting=csv.QUOTE_NONE)

identifier_tuple, syn_vectors = synsets_vectorized(emb=model, worded_synsets=synset_words, named_synsets=synsets_names, tags=TAGS, pos=POS)
print('Number of vectorised synsets', len(syn_vectors))

counter = 0
nosamename = 0
pred_dict = defaultdict(list)
for hypo, hyper_vec in zip(test, hyper_vecs):
    # print(hypo)
    if not np.any(hyper_vec):
        for line in top_ten: # synset ids already
            row = [hypo.strip(), line.strip(), 'dummy']
            writer.writerow(row)
            pred_dict[hypo.strip()].append(line.strip())
    else:
        if effort == 'base':
            # (default) list of (synset_id, hypernym_word, sim)
            this_hypo_res = lemmas_based_hypers(hypo, vec=hyper_vec, emb=model, topn=TOPN, dict_w2ids=synsets_dict)
        if effort == 'averaged_syns':
        # list of (synset_id, hypernym_synset_name, sim)
            this_hypo_res = mean_synset_based_hypers(hypo, vec=hyper_vec, syn_ids=identifier_tuple, syn_vecs=syn_vectors, topn=10)
    
        if counter % 300 == 0:
            print('\n%d hyponyms processed out of %d total' % (counter, len(test)),
                  file=sys.stderr)
            # Want to see predictions in real time?
            print(
                'Here comes a list of 10 unique synset ids for %s in test.\n '
                '%s' % (hypo, this_hypo_res))
        
        counter += 1
        
        for line in this_hypo_res:
            row = [hypo.strip(), line[0].strip(), line[1].strip()]
            writer.writerow(row)
            pred_dict[hypo.strip()].append(line[0].strip())
            
json.dump(pred_dict, open('%s%s_%s_pred_%s.json' % (OUT, POS, TEST, effort), 'w'))
# print('Number of samename hypernyms excluded',nosamename)
outfile.close()

if TEST == 'provided':
    print('===Look at OOV===')
    print('АНИСОВКА', [synsets_names[name] for name in pred_dict['АНИСОВКА']])
    print(pred_dict['ВЭЙП'])
    print(pred_dict['ДРЕСС-КОД'])
    # upload this archive to the site
    archive_name = '%s_%s_%s_%s_%s_%s_%s.zip' % (VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST, effort)
    with zipfile.ZipFile(OUT + archive_name, 'w') as file:
        file.write('%s%s_%s_%s_%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST, effort),
                   '%s_%s_%s_%s_%s_%s_%s.tsv' % (VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST, effort))
    
end = time.time()
training_time = int(end - start)
print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('DONE: %s has run.\n Measuring similarity and formatting output was done in %s minutes' % (os.path.basename(sys.argv[0]),str(round(training_time / 60))))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')