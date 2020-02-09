import os, sys
import argparse
import time
import csv
from smart_open import open
import pandas as pd

from configs import OUT, POS, TEST

# produce a raw train-test split so that test includes only monosemantic hyponym synsets, i.e. synsets that have only one hypernym synset
if TEST == 'provided':
    print('According to the config settings, you are going to use the provided test. Are you sure you want to produce data for intrinsic evaluation?')
    sys.exit()


parser = argparse.ArgumentParser()
if POS == 'NOUN':
    parser.add_argument('--train', default='input/data/training_nouns.tsv', type=os.path.abspath)
if POS == 'VERB':
    parser.add_argument('--train', default='input/data/training_verbs.tsv', type=os.path.abspath)

start = time.time()

args = parser.parse_args()

df_train = pd.read_csv(args.train, sep='\t')

print(df_train.head())
print(df_train.columns)

ids = df_train['SYNSET_ID'].tolist()
my_TEXTS = df_train['TEXT'].tolist()
ids_parents = df_train['PARENTS'].tolist()
my_PARENT_TEXTS = df_train['PARENT_TEXTS'].tolist()

temp = set()


test_size = len(my_TEXTS)*0.2
print('Number of texts to include into test', test_size)


full_test_hypos = []
full_train_hypos = []

for id, hypo, id_par, hyper in zip(ids, my_TEXTS, ids_parents, my_PARENT_TEXTS):
    ## get gid of double quotes inside strings: ПРЕМИЯ "ОСКАР"
    hypo = hypo.replace(r'"', '')
    hyper = hyper.replace(r'"', '')
    if hypo not in temp:
        temp.add(hypo)
        if len(full_test_hypos) < test_size:
            full_test_hypos.append([id, hypo, id_par, hyper])
        else:
            full_train_hypos.append([id, hypo, id_par, hyper])
    else:
        full_train_hypos.append([id, hypo, id_par, hyper])


OUT = '%strains/static/' % OUT
os.makedirs(OUT, exist_ok=True)

print(len(full_train_hypos))
print(len(full_test_hypos))

with open('%s%s_static-train.tsv' % (OUT, POS), 'w') as f:
    writer = csv.writer(f, dialect='unix', delimiter='\t', lineterminator='\n', escapechar='\\', quoting=csv.QUOTE_NONE)
    writer.writerow(['SYNSET_ID', 'TEXT', 'PARENTS', 'PARENT_TEXTS'])
    for row in full_train_hypos:
        writer.writerow(row)

with open('%s%s_static-test.tsv' % (OUT, POS), 'w') as f:
    writer = csv.writer(f, dialect='unix', delimiter='\t', lineterminator='\n',  escapechar='\\', quoting=csv.QUOTE_NONE)
    writer.writerow(['SYNSET_ID', 'TEXT', 'PARENTS', 'PARENT_TEXTS'])
    for row in full_test_hypos:
        writer.writerow(row)

