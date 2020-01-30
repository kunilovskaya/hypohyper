## all you want to know about the input data and resources for DialogueEvaluation2020 Taxonomy Enrichment challenge

from hyper_import_functions import read_xml, read_train, id2wds_dict, wd2id_dict, get_all_rels, get_rel_by_name, id2name_dict
from hyper_import_functions import get_rel_by_synset_id, write_hyp_pairs
import os, sys
import argparse
import time
from xml.dom import minidom

from itertools import repeat

from sklearn.model_selection import train_test_split

## USAGE: python3 temp_explore_input.py --relations /home/u2/resources/hypohyper/ruwordnet/synset_relations.N.xml --synsets /home/u2/resources/hypohyper/ruwordnet/synsets.N.xml --train /home/u2/data/hypohyper/training_data/training_nouns.tsv

RANDOM_SEED = 42
parser = argparse.ArgumentParser()
parser.add_argument('--relations', default='resources/hypohyper/ruwordnet/synset_relations.N.xml', help="synset_relations files from ruwordnet")
parser.add_argument('--synsets', default='resources/hypohyper/ruwordnet/synsets.N.xml', help="synsets files")
parser.add_argument('--senses', default='resources/hypohyper/ruwordnet/senses.N.xml', help="the file with 'main_word' and 'lemma' attibutes to sense tag")
parser.add_argument('--train', default='data/hypohyper/training_data/training_nouns.tsv', help="training_nouns.tsv: SYNSET_ID\tTEXT\tPARENTS\tPARENT_TEXTS")
start = time.time()

args = parser.parse_args()

parsed_syns = read_xml(args.synsets)
parsed_rels = read_xml(args.relations)
doc = minidom.parse(args.senses)
parsed_senses = doc.getElementsByTagName("sense")

get_all_rels(parsed_rels)

## how many synsets have single word members?
print('Total number of senses %d' % len(parsed_senses))

all_id_senses = []
num_synsets = []

uni_ids = []

count_uni = 0
for sense in parsed_senses:
    id = sense.getAttributeNode('synset_id').nodeValue
    name = sense.getAttributeNode("name").nodeValue
    main_wd = sense.getAttributeNode("main_word").nodeValue
    num_synsets.append(id)
    
    if len(name.split()) == 0:
        item = None
        print('Missing name id sense in synset %s' % id)
    elif len(name.split()) > 1:
        item = main_wd
    else:
        count_uni += 1
        item = name
        uni_ids.append(id)
    
    all_id_senses.append((id, item))
        
print(all_id_senses[:20])
print(len(all_id_senses))
print('Ratio of unigram senses to all senses %.2f' % (count_uni/len(parsed_senses)*100))
print('Ratio of synsets that have NO unigram representation %.2f' % (100 - len(set(uni_ids))/len(set(num_synsets))*100))


# df_train = read_train(args.train)
# df_train = df_train.replace(to_replace=r"[\[\]']", value='', regex=True)  ## strip of [] and ' in the strings
#
# my_TEXTS = df_train['TEXT'].tolist()
# my_PARENT_TEXTS = df_train['PARENT_TEXTS'].tolist()
#
# id_dict = id2wds_dict(parsed_syns)
# wd_dict = wd2id_dict(id_dict)
#
# print('=== Testing id2wds ====')
# iterator = iter(id_dict.items())
# for i in range(3):
#     print(next(iterator))
#
# print('=== Testing wd2id ====')
# iterator = iter(wd_dict.items())
# for i in range(3):
#     print(next(iterator))
#
# wd = 'УТКА'
#
# print('=== Testing: Get hyponym ====')
# hypo_wds, hypo_ids = get_rel_by_name(parsed_rels, wd, wd_dict, id_dict, name='hypernym')
# print('All hyponym-synset realizations (=synonyms) for your query:', hypo_wds)
# print('The hyponym-synsets ids for your query:', hypo_ids)
#
# print('=== Testing: Get hypernym ====')
# hyper_wds, hyper_ids = get_rel_by_name(parsed_rels, wd, wd_dict, id_dict, name='hyponym')
# print('All hypernym-synset realizations (=synonyms) for your query:', hyper_wds)
# print('The hypernym-synsets ids for your query:', hyper_ids)
#
# id = '112039-N'
# id_name_dict = id2name_dict(parsed_syns)
# name = id_name_dict[id]
# print('\n%%%%%%%% Get me the name of this synset (%s) stored in ruthes_name attribute: %s\n' % (id,name))
#
# hypo_wds, hypo_ids, this_syn_wds = get_rel_by_synset_id(parsed_rels, id, id_dict, name='hypernym')
# print('Hyponyms for words (synset id %s): %s' % (id, this_syn_wds))
# print(hypo_wds)
#
# all_pairs = []
# noMWE_all_pairs = []
# hyper_count = 0
# tot_pairs = 0
# for hypo, hyper in zip(my_TEXTS, my_PARENT_TEXTS):
#     hypo = hypo.split(', ')
#     # print('===', hypo)
#     hyper = hyper.split(', ')
#     # print('+++', hyper)
#     this_syn = len(hyper)
#     hyper_count += this_syn
#     for i in hypo:  ## in top 5 synsets there are 13 hyponyms and
#         wd_tuples = list(zip(repeat(i), hyper))
#         tot_pairs += len(wd_tuples)
#         all_pairs.append(wd_tuples)
# all_pairs = [item for sublist in all_pairs for item in sublist]  ## flatten the list
# print('======', all_pairs[:5])
# print('Checksum: expected  %d; returned: %d' % (tot_pairs, len(all_pairs)))
#
# for (hypo, hyper) in all_pairs:
#     if not ' ' in hypo and not ' ' in hyper:
#         noMWE_all_pairs.append((hypo, hyper))
# print('Filtered examples:\n', noMWE_all_pairs[:5])
# print('\nFiltered for MWE train: %d pairs' % len(noMWE_all_pairs))
#
# hypohyper_train, hypohyper_test = train_test_split(noMWE_all_pairs, test_size=.5, random_state=RANDOM_SEED)
#
# print(len(hypohyper_train))
# print(len(hypohyper_test))
#
# write_hyp_pairs(hypohyper_train, '/home/u2/data/hypohyper/noMWE_hypohyper_train.txt')
# write_hyp_pairs(hypohyper_test, '/home/u2/data/hypohyper/noMWE_hypohyper_test.txt')
#
# end = time.time()
# processing_time = int(end - start)
# print('Time spent exploring inputs: %.2f minites' % (processing_time / 60), file=sys.stderr)
