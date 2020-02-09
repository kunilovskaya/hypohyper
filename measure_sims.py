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

from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, TAGS, TOPN, OPT, TEST
from hyper_imports import popular_generic_concepts, load_embeddings, parse_taxonymy

parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
# for ultimate results to submit use private instead of public
if TEST == 'provided':
    if POS == 'NOUN':
        parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
if TEST == 'intrinsic':
    if POS == 'NOUN':
        parser.add_argument('--test', default='%strains/%s_%s_test4testing.txt' % (OUT, VECTORS, POS), type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='%strains/%s_%s_test4testing.txt', type=os.path.abspath)
    parser.add_argument('--truth', default='%strains/%s_%s_%s_test.tsv.gz' % (OUT, VECTORS, POS, TEST))
parser.add_argument('--hyper_vectors', default='%spredicted_hypers/%s_%s_%s_%s_hypers.npy' % (OUT, VECTORS, POS, OPT, TEST),
                    help="predicted vectors")

args = parser.parse_args()

start = time.time()

if POS == 'NOUN':
    senses = '%ssenses.N.xml' % RUWORDNET
elif POS == 'VERB':
    senses = '%ssenses.V.xml' % RUWORDNET
else:
    senses = None
    print('Not sure which PoS-domain you want from ruWordNet')
    
print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

sens_index = parse_taxonymy(senses, tags=TAGS, pos=POS, mode=MODE, emb_voc=model.vocab)

synsets_dict = defaultdict(set)

for i in sens_index:
    synset = i[0]
    lemma = i[1]
    synsets_dict[lemma].add(synset)


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
outfile = open('%s%s_%s_%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST), 'w')
writer = csv.writer(outfile, dialect='unix', delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE)

counter = 0

pred_dict = defaultdict(list)
for hypo, hyper_vec in zip(test, hyper_vecs):
    # print(hypo)
    if not np.any(hyper_vec):
    # if len(hyper_vec) == 1:
        # this happens if your OOV_STRATEGY == 'top_hyper';
        # in this case use the sorted list of 10 most frequent hypernyms
        # for each of 10 lines write a triplet = get your required 10 dummy answers!
        for line in top_ten:
            row = [hypo.strip(), line.strip(), 'dummy']
            writer.writerow(row)
            pred_dict[hypo.strip()].append(line.strip())
            # print('OOV', line.strip()) # synset ids already
    else:
        hyper_vec = np.array(hyper_vec, dtype=float)
        temp = set()
        deduplicated_sims = []
        nearest_neighbors = model.most_similar(positive=[hyper_vec], topn=TOPN)
        sims = []
        for res in nearest_neighbors:
            word = res[0]
            similarity = res[1]
            if word in synsets_dict:
                for synset in synsets_dict[word]:
                    sims.append((synset, word, similarity))
        
        # sort the list of tuples (id, sim) by the 2nd element and deduplicate
        # by rewriting the list while checking for duplicate synset ids
        sims = sorted(sims, key=itemgetter(2), reverse=True)
        
        for a, b, c in sims:
            if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b, c))
        this_hypo_res = deduplicated_sims[:10]  # list of (synset_id, hypernym_word, sim)
        
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

outfile.close()

archive_name = '%s_%s_%s_%s_%s_%s_res.zip' % (VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST)
with zipfile.ZipFile(OUT + archive_name, 'w') as file:
    file.write('%s%s_%s_%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST),
               '%s_%s_%s_%s_%s_%s.tsv' % (VECTORS, POS, MODE, OOV_STRATEGY, OPT, TEST))
    
end = time.time()
training_time = int(end - start)

print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('DONE: %s has run.\n Measuring similarity and formatting output was done in %s minutes' % (os.path.basename(sys.argv[0]),str(round(training_time / 60))))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

if TEST == 'intrinsic':
    print('Evaluating on the intrinsic testset')

    gold_dict = defaultdict(set)
    gold = open(args.truth, 'r').readlines()
    for id,line in enumerate(gold):
        # skip the header
        if id == 0:
            continue
            
        # intrinsic test is formated as wd:1067-N, wd:1067-N]
        hypo = line.split('\t')[0].strip()
        hypo = hypo[:-5].upper()
        hyper_id = line.split('\t')[1].strip()
    
        gold_dict[hypo].add(hyper_id)
        
    mean_ap, mean_rr = get_score(gold_dict, pred_dict)
    print("MAP: {0}\nMRR: {1}\n".format(mean_ap, mean_rr), file=sys.stderr)

    print(list(gold_dict.keys())[:5])
    

    print(len(gold_dict), len(pred_dict))