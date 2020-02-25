import argparse
import sys, os
import time
from hyper_imports import wd2id_dict, id2wds_dict, read_xml, get_orgtest, get_orgtrain
from hyper_imports import load_embeddings, preprocess_wordpair
from configs import VECTORS, EMB_PATH, OUT, POS, TAGS, MWE, SKIP_OOV, METHOD, TEST, RUWORDNET

import json

parser = argparse.ArgumentParser()
if TEST == 'provided':
    if POS == 'NOUN':
        parser.add_argument('--train', default='input/data/org_split/train_nouns.tsv', type=os.path.abspath)
        parser.add_argument('--dev', default='input/data/org_split/dev_nouns.tsv', type=os.path.abspath)
        parser.add_argument('--test', default='input/data/org_split/test_nouns.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--train', default='input/data/org_split/train_verbs.tsv', type=os.path.abspath)
        parser.add_argument('--dev', default='input/data/org_split/dev_verbs.tsv', type=os.path.abspath)
        parser.add_argument('--test', default='input/data/org_split/test_nouns.tsv')
else:
    print('Select provided train-dev-test split', file=sys.stderr)
args = parser.parse_args()

start = time.time()

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

if METHOD != 'deworded':  # or METHOD != 'deworded1'
    
    if POS == 'NOUN':
        synsets = '%ssynsets.N.xml' % RUWORDNET
    elif POS == 'VERB':
        synsets = '%ssynsets.V.xml' % RUWORDNET
    else:
        synsets = None
        print('Not sure which PoS-domain you want from ruWordNet', file=sys.stderr)
    
    parsed_syns = read_xml(synsets)
    synset_words = id2wds_dict(parsed_syns)  # a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]
    
    word_syn_ids = wd2id_dict(synset_words)  # ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]

OUT_TRAIN = '%strains/' % OUT
os.makedirs(OUT_TRAIN, exist_ok=True)
OUT_GOLD = '%sgold_dicts/' % OUT
os.makedirs(OUT_GOLD, exist_ok=True)

print('METHOD == %s' % METHOD, file=sys.stderr)
print('TEST == %s' % TEST, file=sys.stderr)
print('Datasets are NOT filtered for OOV or lowercased/tagged just yet, except MWE are deleted in tests',
      file=sys.stderr)

hypohyper_train = get_orgtrain(args.train, map=synset_words)
hypohyper_dev = get_orgtrain(args.dev, map=synset_words)
hypohyper_test = get_orgtrain(args.test, map=synset_words)

gold_dict = get_orgtest(args.test)  # hypos are filtered for MWE

first3pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:3]}
with open('lists/%s_%s_WORDS.txt' % (POS, TEST), 'w') as my_testfile:
    for key in gold_dict:
        my_testfile.write(key + '\n')
        
print('Raw train pairs: %d' % len(hypohyper_train), file=sys.stderr)
print('RAW TRAIN: ', hypohyper_train[:3], file=sys.stderr)
print('Raw dev pairs: %d' % len(hypohyper_dev), file=sys.stderr)
print('RAW DEV: ', hypohyper_dev[:3], file=sys.stderr)
print('Test words (no MWE): %d' % len(gold_dict), file=sys.stderr)
print('GOLD:', first3pairs_gold, file=sys.stderr)
print()

# lists of tuples
train = preprocess_wordpair(hypohyper_train, tags=TAGS, mwe=MWE, pos=POS)
dev = preprocess_wordpair(hypohyper_dev, tags=TAGS, mwe=MWE, pos=POS)
test = preprocess_wordpair(hypohyper_test, tags=TAGS, mwe=MWE, pos=POS)

print('Preprocessed train:', train[:3])
print('Preprocessed dev:', dev[:3])

json.dump(train, open('%s%s_%s_%s_%s_train.json' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w'))
json.dump(dev, open('%s%s_%s_%s_%s_dev.json' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w'))
json.dump(test, open('%sGOLDWORDS_%s_%s_%s_%s.json' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w'))

end = time.time()
training_time = int(end - start)
print('=== %s has run ===' % (os.path.basename(sys.argv[0])))

