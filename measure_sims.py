import argparse
import csv
import os
import sys
import time
import zipfile
from operator import itemgetter
import numpy as np

from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, TAGS, TOPN
from hyper_imports import popular_generic_concepts, load_embeddings, parse_taxonymy

parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
# for ultimate results to submit use private instead of public
if POS == 'NOUN':
    parser.add_argument('--provided_test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
if POS == 'VERB':
    parser.add_argument('--provided_test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
parser.add_argument('--hyper_vectors', default='%spredicted_hypers/%s_%s_hyper_collector.npy' % (OUT, VECTORS, POS),
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

synsets_dict = {}

for i in sens_index:
    synset = i[0]
    lemma = i[1]
    if lemma not in synsets_dict:
        synsets_dict[lemma] = set()
    synsets_dict[lemma].add(synset)
    
# print(synsets_dict)

if OOV_STRATEGY == 'top_hyper':
    if POS == 'NOUN':
        # print('&&&&&&&&&&&&')
        rel_path = '%ssynset_relations.N.xml' % RUWORDNET
    elif POS == 'VERB':
        rel_path = '%ssynset_relations.V.xml' % RUWORDNET
    else:
        rel_path = None
        print('Which PoS?')
    top_ten = popular_generic_concepts(rel_path)
else:
    top_ten = None

test = [i.strip() for i in open(args.provided_test, 'r').readlines()]

hyper_vecs = np.load(args.hyper_vectors, allow_pickle=True)

# if len(test) == len(hyper_vecs):
#     print(
#         'Checking lengths for lists of hyponyms and corresponding hypernym vectors: passed, %d word-vector pairs' % len(
#             test))
# else:
#     print(len(test), len(hyper_vecs))

# for word, measure cosine similarity from all (single word) sense vectors present in embeddings
# to the word's hypernym vector
OUT = '%sresults/' % OUT
os.makedirs(OUT, exist_ok=True)

outfile = open('%s%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY), 'w')

writer = csv.writer(outfile, dialect='unix', delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE)

counter = 0

for hypo, hyper_vec in zip(test, hyper_vecs):
    if len(hyper_vec) == 1:
        # this happens if your OOV_STRATEGY == 'top_hyper';
        # in this case return the sorted list of 10 most frequent hypernyms
        # print('================ WHY am I here?', hypo)
        for line in top_ten:
            row = [hypo.strip(), line.strip(), 'dummy']
            writer.writerow(row)
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
                'One of the vectors (representing its lemmas) is found most similar to %s hypernym vector '
                'produced by the projection model:\n%s' % (hypo, hypo, this_hypo_res))
        
        counter += 1
        
        for line in this_hypo_res:
            row = [hypo.strip(), line[0].strip(), line[1].strip()]
            writer.writerow(row)

outfile.close()

archive_name = '%s_%s_%s_%s.zip' % (VECTORS, POS, MODE, OOV_STRATEGY)
with zipfile.ZipFile(OUT + archive_name, 'w') as file:
    file.write('%s%s_%s_%s_%s.tsv' % (OUT, VECTORS, POS, MODE, OOV_STRATEGY),
               '%s_%s_%s_%s.tsv' % (VECTORS, POS, MODE, OOV_STRATEGY))

end = time.time()
training_time = int(end - start)

print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('DONE: final results, step 5.\n Measuring similarity and formatting output was done in %s minutes' % str(
    round(training_time / 60)))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
