import argparse
import time
import os
import sys
from hyper_imports import process_tsv, process_tsv_deworded_hypers
from hyper_imports import wd2id_dict, id2wds_dict, read_xml, get_orgtest_deworded, get_orgtrain_deworded, get_orgtrain
import json
from configs import VECTORS, RUWORDNET, OUT, POS, TEST, METHOD

## postpone filtering thru embedding and respective CAPS conversion and TAGS-attaching to learning and testing time

parser = argparse.ArgumentParser()

if TEST == 'codalab':
    if POS == 'NOUN':
        parser.add_argument('--train', default='input/data/training_nouns.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--train', default='input/data/training_verbs.tsv', type=os.path.abspath)
if TEST == 'provided':
    if POS == 'NOUN':
        parser.add_argument('--train', default='input/data/org_split/training_nouns.tsv', type=os.path.abspath)
        parser.add_argument('--test', default='input/data/org_split/test_nouns.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--train', default='input/data/org_split/training_verbs.tsv', type=os.path.abspath)
        parser.add_argument('--test', default='input/data/org_split/test_nouns.tsv')
   
start = time.time()

args = parser.parse_args()

synset_words = None
word_syn_ids = None

if METHOD != 'deworded':
        
    if POS == 'NOUN':
        synsets = '%ssynsets.N.xml' % RUWORDNET
    elif POS == 'VERB':
        synsets = '%ssynsets.V.xml' % RUWORDNET
    else:
        synsets = None
        print('Not sure which PoS-domain you want from ruWordNet', file=sys.stderr)
    
    parsed_syns = read_xml(synsets)
    synset_words = id2wds_dict(parsed_syns) # a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]
    
    word_syn_ids = wd2id_dict(synset_words) # ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]


OUT_TRAIN = '%strains/' % OUT
os.makedirs(OUT_TRAIN, exist_ok=True)
OUT_GOLD = '%sgold_dicts/' % OUT
os.makedirs(OUT_GOLD, exist_ok=True)

print('METHOD == %s' % METHOD, file=sys.stderr)
print('TEST == %s' % TEST, file=sys.stderr)
print('Datasets are NOT filtered for OOV or lowercased/tagged just yet, except MWE are deleted in tests', file=sys.stderr)

if METHOD == 'deworded':
    if TEST == 'codalab':
        hypohyper_train = process_tsv_deworded_hypers(args.train)  ## ('ИХТИОЛОГ', '9033-N')
        json.dump(hypohyper_train, open('%s%s_%s_%s_%s_train.json' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w'))
        print('Raw train pairs: %d' % len(hypohyper_train), file=sys.stderr)
 
    elif TEST == 'provided':
        hypohyper_train = get_orgtrain_deworded(args.train)
        json.dump(hypohyper_train, open('%s%s_%s_%s_%s_train.json' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w'))

        gold_dict = get_orgtest_deworded(args.test)  # hypos are filtered for MWE
        json.dump(gold_dict, open('%s%s_%s_%s_gold.json' % (OUT_GOLD, POS, TEST, METHOD), 'w'))
        first3pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:3]}
        with open('%s%s_%s_%s_%s_WORDS.txt' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w') as my_testfile:
            for key in gold_dict:
                my_testfile.write(key + '\n')
        print('Raw train pairs: %d' % len(hypohyper_train), file=sys.stderr)
        print('Test words (no MWE): %d' % len(gold_dict), file=sys.stderr)

        print('RAW TRAIN: ', hypohyper_train[:3], file=sys.stderr)
        print('GOLD:', first3pairs_gold, file=sys.stderr)
        
    else:
        gold_dict = None
        print('Choose the test mode', file=sys.stderr)
    
else:
    if TEST == 'codalab':
        hypohyper_train = process_tsv(args.train)
        json.dump(hypohyper_train, open('%s%s_%s_%s_%s_train.json' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w'))
        print('Raw train pairs: %d' % len(hypohyper_train), file=sys.stderr)
    
    elif TEST == 'provided':
        hypohyper_train = get_orgtrain(args.train, map=synset_words)
        json.dump(hypohyper_train, open('%s%s_%s_%s_%s_train.json' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w'))
    
        gold_dict = get_orgtest_deworded(args.test)  # hypos are filtered for MWE
        json.dump(gold_dict, open('%s%s_%s_%s_gold.json' % (OUT_GOLD, POS, TEST, METHOD), 'w'))
        first3pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:3]}
        with open('%s%s_%s_%s_%s_WORDS.txt' % (OUT_TRAIN, VECTORS, POS, TEST, METHOD), 'w') as my_testfile:
            for key in gold_dict:
                my_testfile.write(key + '\n')
        print('Raw train pairs: %d' % len(hypohyper_train), file=sys.stderr)
        print('Test words (no MWE): %d' % len(gold_dict), file=sys.stderr)
        
        print('RAW TRAIN: ', hypohyper_train[:3], file=sys.stderr)
        print('GOLD:', first3pairs_gold, file=sys.stderr)
    else:
        gold_dict = None
        print('Choose the test mode', file=sys.stderr)

end = time.time()
training_time = int(end - start)
print('=== %s has run ===' % (os.path.basename(sys.argv[0])))
