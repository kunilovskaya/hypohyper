import argparse
import time
from itertools import repeat
import os, re
import sys
from sklearn.model_selection import train_test_split
from hyper_imports import read_xml, read_train, load_embeddings, id2wds_dict, wd2id_dict, \
    filter_dataset, write_hyp_pairs

from configs import VECTORS, TAGS, MWE, EMB_PATH, OUT, RUWORDNET, RANDOM_SEED, POS, SKIP_OOV


parser = argparse.ArgumentParser()
if POS == 'NOUN':
    parser.add_argument('--synsets', default='%ssynsets.N.xml' % RUWORDNET, help="synsets files")
    parser.add_argument('--train', default='input/data/training_nouns.tsv', type=os.path.abspath)
if POS == 'VERB':
    parser.add_argument('--synsets', default='%ssynsets.V.xml' % RUWORDNET, help="synsets files")
    parser.add_argument('--train', default='input/data/training_verbs.tsv', type=os.path.abspath)

start = time.time()

args = parser.parse_args()

parsed_syns = read_xml(args.synsets)
df_train = read_train(args.train)

# strip of [] and ' in the strings:
## TODO maybe average vectors for representatives of each synset in the training_data
df_train = df_train.replace(to_replace=r"[\[\]']", value='', regex=True)

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

my_TEXTS = df_train['TEXT'].tolist()
my_PARENT_TEXTS = df_train['PARENT_TEXTS'].tolist()

id_dict = id2wds_dict(parsed_syns)
wd_dict = wd2id_dict(id_dict)

all_pairs = []

for hypo, hyper in zip(my_TEXTS, my_PARENT_TEXTS):
    hypo = hypo.split(', ')
    hyper = hyper.split(', ')
    this_syn = len(hyper)

    for i in hypo:
        wd_tuples = list(zip(repeat(i), hyper))
        all_pairs.append(wd_tuples)
all_pairs = [item for sublist in all_pairs for item in sublist]  # flatten the list
print('=== Raw training set: %s ===' % len(all_pairs))
print('Raw examples:\n', all_pairs[:3])

# limit training_data to the pairs that are found in the embeddings
filtered_pairs = filter_dataset(all_pairs, model, tags=TAGS, mwe=MWE, pos=POS, skip_oov=SKIP_OOV)
print('\n=== Embeddings coverage: %s ===' % len(filtered_pairs))

print('!!! WYSIWYG as lookup queries!!!')
print('Expecting: TAGS=%s; MWE=%s; %s' % (TAGS, MWE, POS))
print(filtered_pairs[:3])
mwes = [(a,b) for (a,b) in filtered_pairs if re.search('::',a) or re.search('::',b)]
print(mwes[:3])
print('Number of MWE included %s' % len(mwes))

# hypohyper_train, hypohyper_test = train_test_split(filtered_pairs, test_size=.2,
#                                                    random_state=RANDOM_SEED)
hypohyper_train = filtered_pairs

print('Train entries:', len(hypohyper_train), file=sys.stderr)
# print('Test entries:', len(hypohyper_test), file=sys.stderr)

## if any of MWE are in embeddings they look like '::'.join(item.lower().split()) now regardless whether with PoS-tags or without
## this outputs the LOWERCASED words, too

OUT = '%strains/' % OUT
os.makedirs(OUT, exist_ok=True)

write_hyp_pairs(hypohyper_train, '%s%s_%s_train.tsv.gz' % (OUT, VECTORS, POS))
# write_hyp_pairs(hypohyper_test, '%s%s_%s_test.tsv.gz' % (OUT, VECTORS, POS))

end = time.time()
training_time = int(end - start)
print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('DONE formatting train step 0.\n Training data re-formatted in %s minutes' % str(round(training_time/60)))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')