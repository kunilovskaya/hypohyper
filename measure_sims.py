import os, sys
import argparse
from scipy.spatial import distance
from operator import itemgetter
import numpy as np
import csv
import time

# python3 code/hypohyper/measure_sims.py --preprocessed_test proj/hypohyper/output/araneum_preprocessed_test.npy
# --hyper_vectors proj/hypohyper/output/araneum_hyper_collector.npy --sens_vectors proj/hypohyper/output/
# -- mode single_wd --out proj/hypohyper/output/


parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
parser.add_argument('--preprocessed_test', default='output/araneum_preprocessed_test.npy',
                    help='a list of hyponyms to attach to ruWordNet taxonomy', type=os.path.abspath)
parser.add_argument('--hyper_vectors', default='output/araneum_hyper_collector.npy', help="vectors")
parser.add_argument('--sens_vectors', default='proj/hypohyper/output/', help="folder where the output of vectorise_ruwordnet.py is")
parser.add_argument('--mode', default='single_wd', help="if you want to include vectors for main_words in MWE, replace single_wd with main_wd in vectorise_ruwordnet.py and re-run it")
parser.add_argument('--emb_name', default='araneum',
                    help="arbitrary name of the embedding for output formatting purposes: rdt, araneum, cc, other")
parser.add_argument('--out', default='output/', help="where to store the results")

args = parser.parse_args()

start = time.time()

if args.mode == 'single_wd':
    outname = 'araneum_single_ruwordnet_vectorized.npz'
elif args.mode == 'main_wd':
    outname = 'araneum_main_ruwordnet_vectorized.npz'
else:
    print('What do you want to do with sense lexicalised as MWE')
    outname = None

with np.load(args.sens_vectors+outname) as npz:
    sens_index = npz['senses_index']
    sens_vects = npz['senses_vectors']

test = np.load(args.preprocessed_test)
test = [i.strip() for i in test] ## this list of test items; it is UPPER, no tags and filtered for OOV

hyper_vecs = np.load(args.hyper_vectors)

if len(test) == len(hyper_vecs):
    print('Checking lengths for lists of hyponyms and corresponding hypernym vectors: passed, %d word-vector pairs' % len(test))
else:
    print(len(test),len(hyper_vecs))

## for word, measure cosine similarity from all (single word) sense vectors present in embeddings to the word's hypernym vector
outfile = open(args.out+args.emb_name+'_kukupl_results.tsv', 'w')
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
            '\nUnique synset ids for %s in test; one of its members vectors is found most similar to %s hypernym vector produced by the projection model:\n%s' % (hypo, hypo, this_hypo_res))

    counter += 1
    
    for line in this_hypo_res:
        row = [hypo.strip(), line[0].strip(), line[1].strip()]
        writer.writerow(row)
    
outfile.close()

end = time.time()
training_time = int(end - start)
print('Measuring similarity and formatting output was done in %s minutes' % str(round(training_time/60)))

        