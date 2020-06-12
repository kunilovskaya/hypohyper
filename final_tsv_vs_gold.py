"""
compare our output on public test to the gold answered published after the competition

python3 final_tsv_vs_gold.py
"""
import pandas as pd
from collections import defaultdict
from collections import OrderedDict
from xml.dom import minidom
import json
import matplotlib.pyplot as plt
import seaborn as sns

# parse ruwordnet
def read_xml(xml_file):
    doc = minidom.parse(xml_file)
    try:
        parsed = doc.getElementsByTagName("synset") or doc.getElementsByTagName("relation")
    except TypeError:
        # nodelist = parsed
        print('Are you sure you are passing the expected files?')
        parsed = None

    return parsed  # a list of xml entities


def id2wds_dict(synsets):
    id2wds = defaultdict(list)
    for syn in synsets:
        identifier = syn.getAttributeNode('id').nodeValue
        senses = syn.getElementsByTagName("sense")
        for sense in senses:
            wd = sense.childNodes[-1].data
            id2wds[identifier].append(wd)

    return id2wds  # get a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]

def wd2id_dict(id2dict):  # input: id2wds
    wd2ids = defaultdict(list)
    for k, values in id2dict.items():
        for v in values:
            wd2ids[v].append(k)

    return wd2ids  # ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]

def get_score(reference, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in reference:
        reference_hypernyms = reference.get(neologism, [])
        predicted_hypernyms = predicted.get(neologism, [])

        ap_sum += compute_ap(reference_hypernyms, predicted_hypernyms, k)
        # rr_sum += compute_rr(reference_hypernyms, predicted_hypernyms, k)
        rr_sum += compute_rr([j for i in reference_hypernyms for j in i], predicted_hypernyms, k)
    return ap_sum / len(reference), rr_sum / len(reference)


def get_item_score(test, reference, predicted, k=10):
    ap_sum = 0
    rr_sum = 0


    reference_hypernyms = reference.get(test, [])
    predicted_hypernyms = predicted.get(test, [])

    ap_sum += compute_ap(reference_hypernyms, predicted_hypernyms, k)
        # rr_sum += compute_rr(reference_hypernyms, predicted_hypernyms, k)
    rr_sum += compute_rr([j for i in reference_hypernyms for j in i], predicted_hypernyms, k)
    
    return ap_sum, rr_sum


def compute_ap(actual, predicted, k=10):
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    already_predicted = set()
    skipped = 0
    for i, p in enumerate(predicted):
        if p in already_predicted:
            skipped += 1
            continue
        for parents in actual:
            if p in parents:
                num_hits += 1.0
                # counts the score for each component
                score += num_hits / (i + 1.0 - skipped)
                already_predicted.update(parents)
                break

    return score / min(len(actual), k)  # devided by the mumber of commected components


def compute_rr(true, predicted, k=10):
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0)
    return 0.0



gold_dict = defaultdict(list)

gold = 'final_analysis/nouns_public_subgraphs.tsv'
lines = open(gold, 'r').readlines()
temp_dict_ids = defaultdict(list)
temp_dict_wds = defaultdict(list)
org_pairs = []
for line in lines:
    res = line.split('\t')
    wd, ids, hypers = res
    hypers = hypers.replace(r'"', '')  # get rid of dangerous quotes in МАШИНА "ЖИГУЛИ"
    hypers = hypers.replace("'", '"')  # to meet the json requirements
    
    id_list = json.loads(ids)
    hypers_list = json.loads(hypers)
    
    temp_dict_ids[wd].append(id_list)
    temp_dict_wds[wd].append(hypers_list)
components = 0
over_two = 0
for k, v in temp_dict_ids.items():
    if len(v) > 1:
        # print('More than one component', k)
        # print(v)
        components += 1
    if len(v) > 2:
        # print(k)
        # print(v)
        over_two += 1
        
ratio_of_poly = components/len(temp_dict_ids)
print('Ratio of test words with connected components: %1.3f' % ratio_of_poly)
ratio_over_two = over_two/len(temp_dict_ids)
print('Ratio of test words with connected components: %1.3f (%s/%s)' % (ratio_over_two, over_two, len(temp_dict_ids)))
# first2pairs = {k: temp_dict_wds[k] for k in list(temp_dict_wds)[:2]}
# print(first2pairs)

# get maps
synsets = 'input/resources/ruwordnet/synsets.N.xml'
parsed_syns = read_xml(synsets)
id2wd = id2wds_dict(parsed_syns)
wd2id = wd2id_dict(id2wd)

print(id2wd['34-N'])

base_ar = 'final_analysis/w2v-pos-araneum_NOUN_single_top-hyper_codalab_lemmas_raw_anno.tsv'
base = 'final_analysis/mwe-pos-vectors_NOUN_single_top-hyper_codalab-pub_lemmas_raw_none.tsv'
syn = 'final_analysis/mwe-pos-vectors_NOUN_single_top-hyper_codalab-pub_lemmas-neg-syn_raw_none.tsv'
syn_hearst = 'final_analysis/mwe-pos-vectors_NOUN_single_top-hyper_codalab-pub_lemmas-neg-syn_hearst-info50-25_none.tsv'
classif = 'final_analysis/mwe_all_5_386_dev001_overfit_10_synsets_predictions_public.tsv'

# plot the distribution of scores for each system
# systems = [base_ar, base, syn, syn_hearst, classif]
# names = ['base_aranea', 'base', 'syn', 'syn_hearst', 'classif']
# # prepare lists to be added to pd colums
# systnames = []
# scores = []
# freqs = []
# for syst, name in zip(systems, names):
#     preds_dict = defaultdict(list)
#     lines0 = open(syst, 'r').readlines()
#     for line in lines0:
#         res = line.split('\t')
#         wd, id, _= res
#         preds_dict[wd].append(id)
#
#     # scor_dict = {}
#     flipped = defaultdict(list)
#     aggr_scor = defaultdict(list)
#     for i, test_wd in enumerate(preds_dict.keys()):
#         mean_ap, mean_rr = get_item_score(test_wd, temp_dict_ids, preds_dict)
#         mean_ap = round(mean_ap, 3)
#         # scor_dict[test_wd] = mean_ap
#         flipped[mean_ap].append(test_wd)
#         if mean_ap == 0:
#             aggr_scor['0.0'].append(test_wd)
#         elif 0.0 < mean_ap < 0.5:
#
#             aggr_scor['0.0-0.5'].append(test_wd)
#         elif 0.5 < mean_ap < 1.0:
#             print(test_wd, mean_ap)
#             aggr_scor['0.5-1.0'].append(test_wd)
#         elif mean_ap == 1.0:
#             aggr_scor['1.0'].append(test_wd)
#
#     print(aggr_scor.keys())
#     # od = OrderedDict(sorted(aggr_scor.items()))
#     # print('System: %s' % syst.split('/')[1])
#     # print(od.keys())
#     #
#     freq_dict = {}
#     # print('Frequenct distro of scores:')
#     for k, v in aggr_scor.items():
#         print(k, len(v))
#         freq_dict[k] = len(v)
#         systnames.append(name)
#         scores.append(k)
#         freqs.append(len(v))

# thress = pd.DataFrame(columns = ['syst', 'score', 'freq'])
# # 0.0, 0.25, 0.5, 1.0
# thress.syst = systnames
# thress.score = scores
# thress.freq = freqs
# hue_order = ['0.0','0.0-0.5', '0.5-1.0', '1.0']
# sns.set_style("whitegrid")
# sns.set_context('paper')
# sns.barplot(x="syst", y="freq", hue="score", data=thress, hue_order= hue_order, palette=sns.color_palette("cubehelix", 4))
# #вместо серого можно использовать красивую палитру: palette=sns.color_palette("cubehelix", 4)/color="gray"
# plt.xticks(fontsize=14, rotation=0)
# plt.legend(loc='upper right', prop={"size":14})
# plt.ylabel('number of predictions', fontsize=14) #ratio to total no.of hits / relative frequency
# plt.xlabel("", fontsize=14)  # systems
# # plt.despine()
#
# plt.savefig('syst-res_score-freqs.png', format='png')
# plt.show()

preds_dict = defaultdict(list)
system = classif
lines0 = open(system, 'r').readlines()
for line in lines0:
    res = line.split('\t')
    wd, id, _= res
    preds_dict[wd].append(id)
scor_dict = {}
flipped = defaultdict(list)
for i, test_wd in enumerate(preds_dict.keys()):
    mean_ap, mean_rr = get_item_score(test_wd, temp_dict_ids, preds_dict)
    mean_ap = round(mean_ap, 3)
    scor_dict[test_wd] = mean_ap
    flipped[mean_ap].append(test_wd)

# examples of text words with each score
od1 = OrderedDict(sorted(flipped.items()))
# len(preds_dict)
print('System %s:' % system)
for k, v in od1.items():
    print(k, v)
    
print('System %s:' % system)
for i, test_wd in enumerate(preds_dict.keys()):
    # if i < 5:
    # if test_wd in ['АНГЛИЦИЗМ', 'ГЛАГОЛИЦА', 'ГАРЬ', 'ФРАМУГА', 'ОТБЕЛИВАТЕЛЬ', 'КАРАМБОЛЬ']:
    # if test_wd in ['БИЛЬЯРДИСТ', 'ГЕРБАРИЙ', 'ЗАПАСКА', 'ЛЮРЕКС', 'ТУЖУРКА']:
    if test_wd in ['КИТАЕВЕДЕНИЕ', 'АНИСОВКА', 'ХОЛЕРИК', 'СОСНЯК', 'ДИЛЕР', 'МАРАКАС', 'ШПАТЕЛЬ', 'АППЕРКОТ', 'ДУАЛИЗМ', 'АЙПАД']:
        print('\n==\ntestwd: %s' % test_wd)
        print('GOLD ids: %s' % temp_dict_ids[test_wd])
        print('PRED ids: %s' %  preds_dict[test_wd])
        common = set([i for lst in temp_dict_ids[test_wd] for i in lst]).intersection(set(preds_dict[test_wd]))
        print('SHARED: %s' % common)
        mean_ap, mean_rr = get_item_score(test_wd, temp_dict_ids, preds_dict)
        print("MAP: {0:.3f}\nMRR: {1:.3f}\n".format(mean_ap, mean_rr))
        print('GOLD wds: %s' % temp_dict_wds[test_wd])
        print('PRED wds: %s' % [id2wd[x] for x in preds_dict[test_wd]])

mean_ap, mean_rr = get_score(temp_dict_ids, preds_dict, k=10)
print('System %s:' % system)
print("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))
