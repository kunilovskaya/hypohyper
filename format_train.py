import argparse
import time
from itertools import repeat
import os
import sys
from sklearn.model_selection import train_test_split
from hyper_import_functions import read_xml, read_train, load_embeddings, id2wds_dict, wd2id_dict, \
    filter_dataset, write_hyp_pairs

# USAGE:
# python3 format_train.py --relations input/resources/ruwordnet/synset_relations.N.xml
# --synsets input/resources/ruwordnet/synsets.N.xml --train input/data/training_nouns.tsv
# --emb /home/user/model0.model


RANDOM_SEED = 42
parser = argparse.ArgumentParser()
parser.add_argument('--relations', default='input/resources/ruwordnet/synset_relations.N.xml',
                    help="synset_relations files from ruwordnet", type=os.path.abspath)
parser.add_argument('--synsets', default='input/resources/ruwordnet/synsets.N.xml',
                    help="synsets files", type=os.path.abspath)
parser.add_argument('--train', default='input/data/training_nouns.tsv',
                    help="training_nouns.tsv: SYNSET_ID\tTEXT\tPARENTS\tPARENT_TEXTS",
                    type=os.path.abspath)
parser.add_argument('--tags', dest='tags', action='store_true', help="POS tags in embeddings?")
parser.add_argument('--mwe', dest='mwe', action='store_true', help="MWE in embeddings?")
parser.add_argument('--emb',
                    help="path to embeddings file, pay attention to tags and mwe to "
                         "True/False accordingly")

start = time.time()

parser.set_defaults(tags=False)
parser.set_defaults(mwe=True)
args = parser.parse_args()

parsed_syns = read_xml(args.synsets)
parsed_rels = read_xml(args.relations)

df_train = read_train(args.train)

# strip of [] and ' in the strings:
df_train = df_train.replace(to_replace=r"[\[\]']", value='', regex=True)

print('Datasets loaded', file=sys.stderr)

emb_path = args.emb

emb = load_embeddings(emb_path)
# vec = get_vector('дом_NOUN', emb=model)
# print('%%%%%%%%%%%%%%%%%%%%%', len(vec))

my_TEXTS = df_train['TEXT'].tolist()
my_PARENT_TEXTS = df_train['PARENT_TEXTS'].tolist()

id_dict = id2wds_dict(parsed_syns)
wd_dict = wd2id_dict(id_dict)

all_pairs = []
noMWE_all_pairs = []
hyper_count = 0
tot_pairs = 0
for hypo, hyper in zip(my_TEXTS, my_PARENT_TEXTS):
    hypo = hypo.split(', ')
    hyper = hyper.split(', ')
    this_syn = len(hyper)
    hyper_count += this_syn
    for i in hypo:
        wd_tuples = list(zip(repeat(i), hyper))
        tot_pairs += len(wd_tuples)
        all_pairs.append(wd_tuples)
all_pairs = [item for sublist in all_pairs for item in sublist]  # flatten the list
print('======', all_pairs[:5])
print('Checksum: expected  %d; returned: %d' % (tot_pairs, len(all_pairs)))

# for (hypo, hyper) in all_pairs:
#     if not ' ' in hypo and not ' ' in hyper:
#         noMWE_all_pairs.append((hypo, hyper))
# print('Filtered examples:\n', noMWE_all_pairs[:5])
# print('\nFiltered for MWE train: %d pairs' % len(noMWE_all_pairs))


# limit to the pairs that are found in the embeddings
filtered_pairs = filter_dataset(all_pairs, emb, tags=args.tags, mwe=args.mwe)
print('Number of word pairs where both items are in embeddings:', len(filtered_pairs))

print(filtered_pairs[:3])

hypohyper_train, hypohyper_test = train_test_split(filtered_pairs, test_size=.2,
                                                   random_state=RANDOM_SEED)

print('Train entries:', len(hypohyper_train), file=sys.stderr)
print('Test entries:', len(hypohyper_test), file=sys.stderr)

write_hyp_pairs(hypohyper_train, 'outputs/%s-upos_hypohyper_train.tsv.gz' % args.emb.split('/')[-1])
write_hyp_pairs(hypohyper_test, 'outputs/%s-upos_hypohyper_test.tsv.gz' % args.emb.split('/')[-1])
