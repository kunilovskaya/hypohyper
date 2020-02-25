import argparse
import csv
import os
import sys
import time
import zipfile
import numpy as np
from smart_open import open
from collections import defaultdict
import json

from configs import VECTORS, OUT, RUWORDNET, OOV_STRATEGY, POS, MODE, EMB_PATH, FT_EMB, TAGS, vecTOPN, \
    TEST, METHOD, FILTER_1, FILTER_2
from hyper_imports import popular_generic_concepts, load_embeddings, filtered_dicts_mainwds_option,read_xml, id2name_dict
from hyper_imports import lemmas_based_hypers, mean_synset_based_hypers, synsets_vectorized, disambiguate_hyper_syn_ids
from hyper_imports import cooccurence_counts, preprocess_mwe, lose_family_anno, lose_family_comp, get_generations, map_mwe

parser = argparse.ArgumentParser('Detecting most similar synsets and formatting the output')
# for ultimate results to submit use private instead of public
if TEST == 'codalab-pub':
    if POS == 'NOUN':
        parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
if TEST == 'codalab-pr':
    if POS == 'NOUN':
        parser.add_argument('--test', default='input/data/private_test/nouns_private.tsv', type=os.path.abspath)
    if POS == 'VERB':
        parser.add_argument('--test', default='input/data/private_test/verbs_private.tsv', type=os.path.abspath)
        
if TEST == 'provided':
        parser.add_argument('--test', default='lists/%s_%s_WORDS.txt' % (POS, TEST), type=os.path.abspath)

if FILTER_1 == 'comp':
    if POS == 'NOUN':
        parser.add_argument('--train', default='input/data/training_nouns.tsv',
                            help="train data in format SYNSET<TAB>SENSES<TAB>PARENTS<TAB>DEFINITION")
    if POS == 'VERB':
        parser.add_argument('--train', default='input/data/training_verbs.tsv',
                            help="train data in format SYNSET<TAB>SENSES<TAB>PARENTS<TAB>DEFINITION")
    ## is lemmas-neg-hyp add _lambda05
parser.add_argument('--hyper_vectors', default='%spredicted_hypers/%s_%s_%s_%s_%s_hypers_lambda05.npy' % (OUT, VECTORS, POS,
                                                                                                 OOV_STRATEGY,
                                                                                                 TEST, METHOD),
                    help="predicted vectors")

args = parser.parse_args()

start = time.time()
    
print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
model = load_embeddings(EMB_PATH)

if FILTER_1 == 'disamb':
    print('FT embedding model:', FT_EMB.split('/')[-1], file=sys.stderr)
    ft_emb = load_embeddings(FT_EMB)
    
if 'corp-info' in FILTER_1 or 'hearst-info' in FILTER_1:
    bits = FILTER_1.split('-')
    COOC_LIMIT = int(bits[-1])  # int(''.join([i for i in FILTER_1 if i.isdigit()]))
    DEDUP_LIMIT = int(bits[-2][-2:])

if POS == 'NOUN':
    senses = '%ssenses.N.xml' % RUWORDNET
    synsets = '%ssynsets.N.xml' % RUWORDNET
elif POS == 'VERB':
    senses = '%ssenses.V.xml' % RUWORDNET
    synsets = '%ssynsets.V.xml' % RUWORDNET
else:
    senses = None
    synsets = None
    print('Not sure which PoS-domain you want from ruWordNet')
    
parsed_syns = read_xml(synsets)
id2name = id2name_dict(parsed_syns)
## provide for extented MWE preprocessing in process_mwe
if VECTORS  == 'mwe-vectors':
    source = open('%smwe/ruWordNet_names.txt' % OUT, 'r').readlines()
    source_tagged = open('%s/mwe/ruWordNet_same-names_pos.txt' % OUT, 'r').readlines()
    mwe_map = map_mwe(names=source, same_names=source_tagged)
else:
    mwe_map = None
# this is where single/main word MODE is applied
## get {'родитель_NOUN': ['147272-N', '136129-N', '5099-N', '2655-N'], 'злоупотребление_NOUN': ['7331-N', '117268-N'...]}
## and its reverse
lemmas2ids, id2lemmas = filtered_dicts_mainwds_option(senses, tags=TAGS, pos=POS, mode=MODE, emb_voc=model.vocab, map=mwe_map)

identifier_tuple, syn_vectors = synsets_vectorized(emb=model, id2lemmas=id2lemmas,
                                                   named_synsets=id2name, tags=TAGS, pos=POS)
print('Number of vectorised synsets %d out of 29297; checksums with identifiers (%d)' % (len(syn_vectors), len(identifier_tuple)))

if OOV_STRATEGY == 'top-hyper' or FILTER_1 == 'anno':
    if POS == 'NOUN':
        rel_path = '%ssynset_relations.N.xml' % RUWORDNET
    elif POS == 'VERB':
        rel_path = '%ssynset_relations.V.xml' % RUWORDNET
    else:
        rel_path = None
        print('Which PoS?')
    top_ten = popular_generic_concepts(rel_path)
    if FILTER_1 == 'anno':
        print('Why do I get here?')
        rel_lookup = get_generations(rel_path, redundant=FILTER_2)
else:
    top_ten = None

test = [i.strip() for i in open(args.test, 'r').readlines()]

hyper_vecs = np.load(args.hyper_vectors, allow_pickle=True)

OUT_RES = '%sresults/' % OUT
os.makedirs(OUT_RES, exist_ok=True)

if 'codalab' in TEST:
    outfile = open('%s%s_%s_%s_%s_%s_%s_%s_%s.tsv' % (OUT_RES, VECTORS, POS, MODE, OOV_STRATEGY,
                                                      TEST, METHOD, FILTER_1, FILTER_2), 'w')
    writer = csv.writer(outfile, dialect='unix', delimiter='\t', lineterminator='\n', escapechar='\\', quoting=csv.QUOTE_NONE)


counter = 0
pred_dict = defaultdict(list)
pred_dict_lemmas = defaultdict(list)

monosem = 0
polyN = 0
tot_hypernyms = 0
for hypo, hyper_vec in zip(test, hyper_vecs):
    if not np.any(hyper_vec):
        for line in top_ten: # synset ids already
            row = [hypo.strip(), line.strip(), 'dummy']
            if 'codalab' in TEST:
                writer.writerow(row)
            pred_dict[hypo.strip()].append(line.strip())
    else:
        if METHOD == 'lemmas' or METHOD == 'lemmas-neg-hyp' or METHOD == 'lemmas-neg-syn':
            # (default) get a list of (synset_id, hypernym_word, sim)
            # dict_w2ids = {'родитель_NOUN': ['147272-N', '136129-N', '5099-N', '2655-N']}
            
            item = preprocess_mwe(hypo, tags=TAGS, pos=POS)
            ## this limit is the upperbound of the limit within which we are re-ordering predicted hypers
            deduplicated_res = lemmas_based_hypers(item, vec=hyper_vec, emb=model, topn=vecTOPN, dict_w2ids=lemmas2ids, limit=50)
            # print('This test item output is deduplicated')
            # use FILTER disamb to retain only one, most similar component of polysemantic hypernyms, instead of grabbing the first one
            if FILTER_1 == 'disamb': # <- list of [(id1_1,hypernym1), (id1_2,hypernym1), (id2_1,hypernym2), (id2_2,hypernym2)]
                # list of (one_id, hypernym_word) and stats
                relevant_ids, one_comp, overN, tot_hypers = disambiguate_hyper_syn_ids(item,
                                                                                       list_to_filter=deduplicated_res,
                                                                                       emb=model, ft_model=ft_emb,
                                                                                       index_tuples=identifier_tuple,
                                                                                       mean_syn_vectors=syn_vectors)
                monosem += one_comp
                polyN += overN
                tot_hypernyms += tot_hypers
                
                this_hypo_res = relevant_ids[:10]
                
            elif FILTER_1 == 'anno': # <- list of [(id1_1,hypernym1), (id1_2,hypernym1), (id2_1,hypernym2), (id2_2,hypernym2)]
                ## TASK: exclude words that have parents in predictions for this test word;
                # return a smaller (id,hyper_word) list
                norelatives = lose_family_anno(hypo, deduplicated_res, rel_lookup) ## FILTER_2 is applied above
                this_hypo_res = norelatives[:10]
                
            elif FILTER_1 == 'comp':  # <- list of [(id1_1,hypernym1), (id1_2,hypernym1), (id2_1,hypernym2), (id2_2,hypernym2)]
                ## same TASK, pretends to take into account connected components
                norelatives = lose_family_comp(hypo, deduplicated_res, train=args.train, redundant=FILTER_2)
                this_hypo_res = norelatives[:10]
            
            elif 'corp-info' in FILTER_1:
                ## load the lists of hypernyms that coocur with the given hyponyms
                freqs_dict = json.load(open('%scooc/merged5corp_%s_freq_cooc50_%s.json' % (OUT, TEST, POS), 'r'))
                ## last options influence the performance: they regulate how much fredom of upward movement we allow for coocuring items
                # print(COOC_LIMIT, DEDUP_LIMIT)
                # print('Co-occurences for %d items' % len(freqs_dict))
                cooc_updated = cooccurence_counts(hypo, deduplicated_res,
                                                   corpus_freqs=freqs_dict, thres_cooc=COOC_LIMIT, thres_dedup=DEDUP_LIMIT)
                this_hypo_res = cooc_updated[:10]
                
            elif 'hearst-info' in FILTER_1:
                hearst_dict = json.load(open('%scooc/hearst-hypers_merged-news-taxonomy-ruscorpwiki-rncP-pro_%s_%s.json' % (OUT,POS,TEST), 'r'))
                
                hearst_updated = cooccurence_counts(hypo, deduplicated_res,
                                                  corpus_freqs=hearst_dict, thres_cooc=COOC_LIMIT,
                                                  thres_dedup=DEDUP_LIMIT)
                this_hypo_res = hearst_updated[:10]
            else:
                this_hypo_res = deduplicated_res[:10]
                
        elif METHOD == 'deworded':
            ## gets a list of (synset_id, hypernym_synset_NAME, sim); identifier_tuple is 134530-N, КУНГУР
            this_hypo_res = mean_synset_based_hypers(hypo, vec=hyper_vec, syn_ids=identifier_tuple, syn_vecs=syn_vectors)
            this_hypo_res = this_hypo_res[:10]
            
        else:
            this_hypo_res = None
            print('Any other methods to improve performance?')

        if counter % 500 == 0:
            print('%d hyponyms processed out of %d total' % (counter, len(test)),
                  file=sys.stderr)
            
            ## Want to see predictions in real time?
            # print('%s: %s' % (hypo, this_hypo_res))

        counter += 1

        for line in this_hypo_res:
            row = [hypo.strip(), line[0], line[1]] # (hypo, id, hypernym)
            if 'codalab' in TEST:
                writer.writerow(row)
            pred_dict[hypo.strip()].append(line[0])
            pred_dict_lemmas[hypo.strip()].append(line[1])
            # if hypo in ['ЧИБИС', 'ШАШКА', 'ШАТТЛ', 'ШАТУН', 'ЧАСТУШКА']:
            #     print(hypo, line[1])
if 'codalab' in TEST:
    outfile.close()

if METHOD == 'lemmas' and FILTER_1 == 'disamb':
    print('Total hypernyms processed for %d testwords: %d' % (len(test), tot_hypernyms))
    # print('Monosemantic hypernyms', monosem)
    print('Ratio of monosem hypernyms (disamb is unnecessary): %.2f%%' % (monosem / tot_hypernyms * 100))
    print('Ratio of polysem hypernyms (over 3 components, usefulness of FILTER1): %.2f%%' % (
            polyN / tot_hypernyms * 100))

first3pairs_ids = {k: pred_dict[k] for k in list(pred_dict)[:3]}
# print('PRED_ids:', first3pairs_ids, file=sys.stderr)
first3pairs_hypers = {k: pred_dict_lemmas[k] for k in list(pred_dict_lemmas)[:3]}
# print('PRED_wds:', first3pairs_hypers, file=sys.stderr)

if TEST == 'provided':
    OUT_RES_ORG = '%sorg_split/' % OUT_RES
    os.makedirs(OUT_RES_ORG, exist_ok=True)
    json.dump(pred_dict, open('%s%s_%s_%s_%s_%s_pred.json' % (OUT_RES_ORG, POS, TEST, METHOD, FILTER_1, FILTER_2), 'w'))

elif 'codalab' in TEST:
    print('===Look at %s predictions for OOV===' % OOV_STRATEGY)
    print('АНИСОВКА', [id2name[id] for id in pred_dict['АНИСОВКА']]) ## ВЕЙП, ДРЕСС-КОД
    
    # upload this archive to the site
    archive_name = '%s_%s_%s_%s_%s_%s_%s_%s.zip' % (VECTORS, POS, MODE, OOV_STRATEGY, TEST, METHOD, FILTER_1, FILTER_2)
    with zipfile.ZipFile(OUT_RES + archive_name, 'w') as file:
        file.write('%s%s_%s_%s_%s_%s_%s_%s_%s.tsv' % (OUT_RES, VECTORS, POS, MODE, OOV_STRATEGY, TEST, METHOD, FILTER_1, FILTER_2),
                   '%s_%s_%s_%s_%s_%s_%s_%s.tsv' % (VECTORS, POS, MODE, OOV_STRATEGY, TEST, METHOD, FILTER_1, FILTER_2))

end = time.time()
training_time = int(end - start)

print('=== DONE: %s has run ===\nMeasuring similarity and formatting output was done in %s minutes' %
                                                        (os.path.basename(sys.argv[0]),str(round(training_time / 60))))
