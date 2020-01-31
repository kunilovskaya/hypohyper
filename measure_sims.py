import os, sys
import argparse
from scipy.spatial import distance
from operator import itemgetter
import numpy as np
import csv
import time

from configs import VECTORS, TAGS, MWE, EMB, OUT, RUWORDNET, RANDOM_SEED

vectors = VECTORS
tags = TAGS
mwe = MWE
emb_path = EMB
out = OUT
ruWN = RUWORDNET
RANDOM_SEED = RANDOM_SEED


parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
parser.add_argument('--preprocessed_test', default='%s/%s_preprocessed_test.npy' % (out, vectors),
                    help='a list of hyponyms to attach to ruWordNet taxonomy', type=os.path.abspath)
parser.add_argument('--hyper_vectors', default='%s/%s_hyper_collector.npy' % (out, vectors), help="vectors")
parser.add_argument('--sens_vectors', default=out, help="folder where the output of vectorise_ruwordnet.py is")
parser.add_argument('--mode', default='single_wd', help="if you want to include vectors for main_words in MWE, replace single_wd with main_wd in vectorise_ruwordnet.py and re-run it")

args = parser.parse_args()

start = time.time()

if args.mode == 'single_wd':
    outname = '%s_single_ruwordnet_vectorized.npz' % vectors
elif args.mode == 'main_wd':
    outname = '%s_main_ruwordnet_vectorized.npz' % vectors
else:
    print('What do you want to do with sense lexicalised as MWE')
    outname = None

with np.load(args.sens_vectors+outname) as npz:
    sens_index = npz['senses_index']
    sens_vects = npz['senses_vectors']

test = np.load(args.preprocessed_test)
test = [i.strip() for i in test] ## this list of test items; it is UPPER, no tags and filtered for OOV
print()

hyper_vecs = np.load(args.hyper_vectors)

if len(test) == len(hyper_vecs):
    print('Checking lengths for lists of hyponyms and corresponding hypernym vectors: passed, %d word-vector pairs' % len(test))
else:
    print(len(test),len(hyper_vecs))

## for word, measure cosine similarity from all (single word) sense vectors present in embeddings to the word's hypernym vector
outfile = open(out+vectors+'_kukupl_results.tsv', 'w')
writer = csv.writer(outfile, dialect='unix', delimiter='\t', lineterminator='\n')

counter = 0

for hypo, hyper_vec in zip(test,hyper_vecs):
    hyper_vec = np.array(hyper_vec, dtype=float)
    sims = []
    temp = set()
    deduplicated_sims = []
    for (id, word), vect in zip(sens_index, sens_vects): ## does zip allow set operation?
        vect = np.asarray(vect, dtype='float64')
        sims.append((id, word, (1 - distance.cosine(hyper_vec, vect))))
        
    ## sort the list of tuples (id, sim) by the 2nd element and deduplicate by rewriting the list while checking for duplicate synset ids
    sims = sorted(sims, key=itemgetter(2), reverse=True)

    for a, b, c in sims:
        if not a in temp:
            temp.add(a)
            deduplicated_sims.append((a, b, c))
    this_hypo_res = deduplicated_sims[:10] ## list of (synset_id, hypernym_word, sim)

    if counter % 100 == 0:
        print('%d hyponyms processed out of %d total' % (counter, len(test)),
              file=sys.stderr)
        # Want to see predictions in real time?
        print(
            '\nHere comes a list of 10 unique synset ids for %s in test.\n One of the vectors (representing its lemmas) is found most similar to %s hypernym vector produced by the projection model:\n%s' % (hypo, hypo, this_hypo_res))

    counter += 1
    
    for line in this_hypo_res:
        row = [hypo.strip(), line[0].strip(), line[1].strip()]
        writer.writerow(row)
    
outfile.close()

end = time.time()
training_time = int(end - start)
print('Measuring similarity and formatting output was done in %s minutes' % str(round(training_time/60)))

        