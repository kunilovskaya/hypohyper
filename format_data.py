import argparse
import time
from itertools import repeat
import os, re
import sys
from sklearn.model_selection import train_test_split
from hyper_imports import read_xml, load_embeddings, write_hyp_pairs, process_tsv, process_test_tsv

from configs import VECTORS, TAGS, MWE, EMB_PATH, OUT, RUWORDNET, RANDOM_SEED, POS, SKIP_OOV, TEST


parser = argparse.ArgumentParser()
if POS == 'NOUN':
    parser.add_argument('--synsets', default='%ssynsets.N.xml' % RUWORDNET, help="synsets files")
if POS == 'VERB':
    parser.add_argument('--synsets', default='%ssynsets.V.xml' % RUWORDNET, help="synsets files")
    
if TEST == 'intrinsic':
    if POS == 'NOUN':
        parser.add_argument('--train', default='%strains/static/%s_static-train.tsv' % (OUT, POS), help='static train',
                            type=os.path.abspath)
        parser.add_argument('--test', default='%strains/static/%s_static-test.tsv' % (OUT, POS), help='static test',
                            type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--train', default='%strains/static/%s_static-train.tsv' % (OUT, POS), help='static train',
                            type=os.path.abspath)
        parser.add_argument('--test', default='%strains/static/%s_static-test.tsv' % (OUT, POS), help='static test',
                            type=os.path.abspath)
if TEST == 'provided':
    if POS == 'NOUN':
        parser.add_argument('--train', default='input/data/training_nouns.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--train', default='input/data/training_verbs.tsv', type=os.path.abspath)

   
start = time.time()

args = parser.parse_args()

parsed_syns = read_xml(args.synsets)
print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

OUT = '%strains/' % OUT
os.makedirs(OUT, exist_ok=True)

hypohyper_train = process_tsv(args.train, emb=model, tags=TAGS, mwe=MWE, pos=POS, skip_oov=SKIP_OOV)

print('\n%s train entries: %d\n\n==!!==!!==!!==!!==!!\n' % (TEST.upper(), len(hypohyper_train)), file=sys.stderr)

## if any of MWE are in embeddings they look like '::'.join(item.lower().split()) now regardless whether with PoS-tags or without
## this outputs the LOWERCASED words, too
write_hyp_pairs(hypohyper_train, '%s%s_%s_%s_train.tsv.gz' % (OUT, VECTORS, POS, TEST))
all_pairs = []
total_wds = 0
oov_counter = 0
if TEST == 'intrinsic':
    hypohyper_test = process_test_tsv(args.test, emb=model, tags=TAGS, mwe=MWE, pos=POS, skip_oov=SKIP_OOV)
    print('%s test entries: %d' % (TEST.upper(), len(hypohyper_test)), file=sys.stderr)
    write_hyp_pairs(hypohyper_test, '%s%s_%s_%s_test.tsv.gz' % (OUT, VECTORS, POS, TEST))
    
    with open('%s%s_%s_test4testing.txt' % (OUT, VECTORS, POS), 'w') as my_testfile:
        temp = set()
        for tup in hypohyper_test:
            hypo = tup[0]
            hypo = hypo[:-5].upper()
            if hypo not in temp:
                temp.add(hypo)
                my_testfile.write(hypo + '\n')
            else:
                continue
    print('I have prepared data for intrinsic testing')
    

end = time.time()
training_time = int(end - start)
print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('%s has run' % (os.path.basename(sys.argv[0])))# #\n Training data re-formatted in %s minutes ,str(round(training_time/60))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')