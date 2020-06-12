import csv
import itertools
import json
import logging
import os
import re
import sys
import zipfile
from collections import Counter
from collections import defaultdict
from itertools import repeat
from xml.dom import minidom
import numpy as np
import pandas as pd
from gensim.matutils import unitvec
from gensim.models import KeyedVectors
from smart_open import open
from get_reference_format import get_data
import logging


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


# map synset ids to synset names from synsets.N.xml (ex. ruthes_name="УЛЫБКА")
def id2name_dict(synsets):
    id2name = defaultdict()
    for syn in synsets:
        name = syn.getAttributeNode('ruthes_name').nodeValue
        identifier = syn.getAttributeNode('id').nodeValue
        id2name[identifier] = name

    return id2name


def wd2id_dict(id2dict):  # input: id2wds
    wd2ids = defaultdict(list)
    for k, values in id2dict.items():
        for v in values:
            wd2ids[v].append(k)

    return wd2ids  # ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]


# FYI: distribution of relations annotated in ruwordnet
def get_all_rels(relations):  # <- parsed_rels = read_xml(args.relations) <- synset_relations.N.xml
    my_list = []
    for rel in relations:
        rel_name = rel.getAttributeNode('name').nodeValue
        my_list.append(rel_name)
    my_dict = Counter(my_list)  # Counter({'apple': 3, 'egg': 2, 'banana': 1})

    # iterator = iter(my_dict.items())
    # for i in range(len(my_dict)):
    #     print(next(iterator))
    print('=====\n %s \n=====' % my_dict)


# pass a hyponym to get its hyperonyms (ids)
# name: hypernym, hyponym, domain, POS-synonymy, instance hypernym, instance hyponym:
# relations: # <- parsed_rels = read_xml(args.relations) <- synset_relations.N.xml
def get_rel_by_name(relations, word, wds2ids, id2wds, name=None):
    these_ids = wds2ids[word]  # this is a list of possible ids for this word
    related_ids = []
    related_wds = []

    for rel in relations:
        parent = rel.getAttributeNode('parent_id').nodeValue
        child = rel.getAttributeNode('child_id').nodeValue
        rel_name = rel.getAttributeNode('name').nodeValue
        for this_id in these_ids:
            if this_id == child and name == rel_name:
                related_id = parent
                related_ids.append(related_id)

    for related_syn_id in related_ids:
        related = id2wds[related_syn_id]
        related_wds.append(related)

    return related_wds, related_ids


# relations: # <- parsed_rels = read_xml(args.relations) <- synset_relations.N.xml
# name: hypernym, hyponym, domain, POS-synonymy, instance hypernym, instance hyponym:
def get_rel_by_synset_id(relations, identifier, id2wds, name=None):
    wds_in_this_id = id2wds[identifier]
    related_ids = []
    related_wds = []
    print('Inspecting relations for the synset: %s' % wds_in_this_id)
    for rel in relations:
        parent = rel.getAttributeNode('parent_id').nodeValue
        child = rel.getAttributeNode('child_id').nodeValue
        rel_name = rel.getAttributeNode('name').nodeValue
        if identifier == child and name == rel_name:
            related_id = parent
            related_ids.append(related_id)

    for related_syn_id in related_ids:
        related = id2wds[related_syn_id]
        related_wds.append(related)

    return related_wds, related_ids, wds_in_this_id


def process_tsv(filepath):
    df_train = read_train(filepath)

    df_train = df_train.replace(to_replace=r"[\[\]']", value='', regex=True)

    my_texts = df_train['TEXT'].tolist()
    my_parent_texts = df_train['PARENT_TEXTS'].tolist()

    all_pairs = []

    for hypo, hyper in zip(my_texts, my_parent_texts):
        hypo = hypo.replace(r'"', '')
        hyper = hyper.replace(r'"', '')
        hypo = hypo.split(', ')
        hyper = hyper.split(', ')

        for i in hypo:
            wd_tuples = list(zip(repeat(i), hyper))
            all_pairs.append(wd_tuples)
    all_pairs = [item for sublist in all_pairs for item in sublist]  # flatten the list

    return all_pairs  # ('ИХТИОЛОГ', 'УЧЕНЫЙ')


def process_tsv_deworded_hypers(filepath):
    # open the original training data (or part of it) with json
    lines = open(filepath, 'r').readlines()
    temp_dict = {}
    synset_pairs = []
    for i, line in enumerate(lines):
        # skip the header
        if i == 0:
            continue
        res = line.split('\t')

        _, wds, par_ids, _ = res
        par_ids = par_ids.replace("'", '"')  # to meet the json requirements

        wds = wds.split(', ')  # this column is not a json format!
        for w in wds:
            w = w.replace(r'"', '')  # get rid of dangerous quotes in МАШИНА "ЖИГУЛИ"
            # {'WORD': ['4544-N', '147272-N'], '120440-N': ['141697-N', '116284-N']}
            temp_dict[w] = json.loads(par_ids)

    for hypo_w, hypers_ids in temp_dict.items():
        id_tuples = list(zip(repeat(hypo_w), hypers_ids))
        synset_pairs.append(id_tuples)

    synset_pairs = [item for sublist in synset_pairs for item in sublist]  # flatten the list
    print('Number of wd-to-synset pairs in training_data: ', len(synset_pairs))

    return synset_pairs  # ('ИХТИОЛОГ', '9033-N')


def get_orgtrain_deworded(filepath):
    lines = open(filepath, 'r').readlines()
    temp_dict = defaultdict(list)
    org_pairs = []
    for line in lines:
        res = line.split('\t')
        wd, par_ids = res
        wd = wd.replace(r'"', '')  # get rid of dangerous quotes in МАШИНА "ЖИГУЛИ"
        par_ids = par_ids.replace("'", '"')  # to meet the json requirements
        id_list = json.loads(par_ids)
        for identifier in id_list:
            # {'WORD': [['4544-N'], ['147272-N']], 'WORD': ['141697-N', '116284-N']}
            temp_dict[wd].append(identifier)

    for hypo_w, hypers_ids in temp_dict.items():
        id_tuples = list(zip(repeat(hypo_w), hypers_ids))
        org_pairs.append(id_tuples)

    org_pairs = [item for sublist in org_pairs for item in sublist]  # flatten the list
    print('Number of wd-to-synset pairs in training_data: ', len(org_pairs))

    return org_pairs  # ('ИХТИОЛОГ', '9033-N')


def get_orgtrain(filepath, mapping=None):  # mapping = synset_words
    lines = open(filepath, 'r').readlines()
    temp_dict = defaultdict(list)
    org_pairs = []
    for line in lines:
        res = line.split('\t')
        wd, par_ids = res
        wd = wd.replace(r'"', '')  # get rid of dangerous quotes in МАШИНА "ЖИГУЛИ"
        par_ids = par_ids.replace("'", '"')  # to meet the json requirements
        id_list = json.loads(par_ids)
        for identifier in id_list:
            wd_list = mapping[identifier]
            for i in wd_list:
                temp_dict[wd].append(i)
    for hypo_w, hypers_wds in temp_dict.items():
        wd_tuples = list(zip(repeat(hypo_w), hypers_wds))
        org_pairs.append(wd_tuples)
    org_pairs = [item for sublist in org_pairs for item in sublist]  # flatten the list
    print('Number of wd-to-synset pairs in training_data: ', len(org_pairs))

    return org_pairs  # ('ИХТИОЛОГ', '9033-N')


def get_orgtest(filepath):
    mwe = []
    lines = open(filepath, 'r').readlines()
    gold_dict = defaultdict(list)
    for line in lines:
        res = line.split('\t')
        wd, par_ids = res
        wd = wd.replace(r'"', '')  # get rid of dangerous quotes in МАШИНА "ЖИГУЛИ"

        if len(wd.split()) != 1 or bool(re.search('[a-zA-Z]', wd)):  # skip MWE
            mwe.append(wd)
            continue
        par_ids = par_ids.replace("'", '"')  # to meet the json requirements
        id_list = json.loads(par_ids)
        # {'WORD1': [['4544-N'], ['147272-N']], 'WORD2': [['141697-N', '116284-N']]}
        gold_dict[wd].append(id_list)

    return gold_dict


def read_train(tsv_in):
    df_out = pd.read_csv(tsv_in, sep='\t')
    return df_out


def load_embeddings(modelfile):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if not os.path.isfile(modelfile):
        raise FileNotFoundError("No file called {file}".format(file=modelfile))
    # Determine the model format by the file extension
    # Binary word2vec file:
    if modelfile.endswith('.bin.gz') or modelfile.endswith('.bin') or modelfile.endswith('.w2v'):
        emb_model = KeyedVectors.load_word2vec_format(modelfile, binary=True,
                                                      unicode_errors='replace', limit=3500000)
    elif modelfile.endswith('.txt.gz') or modelfile.endswith('.txt') \
            or modelfile.endswith('.vec.gz') or modelfile.endswith('.vec'):  # Text word2vec file
        emb_model = KeyedVectors.load_word2vec_format(modelfile, binary=False,
                                                      unicode_errors='replace', limit=3500000)
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith('.zip'):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            # metafile = archive.open('meta.json')
            # metadata = json.loads(metafile.read())
            # for key in metadata:
            #    print(key, metadata[key])
            # print('============')
            # Loading the model itself:
            stream = archive.open("model.bin")  # or model.txt, if you want to look at the model
            emb_model = KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:
        # Native Gensim format, inclufing for fasttext models
        # (.model in a folder with the other support files)
        emb_model = KeyedVectors.load(modelfile)
    emb_model.init_sims(replace=True)

    return emb_model


# def get_vector(word, emb=None):
#     if not emb:
#         return None
#     vector = emb[word]
#     return vector
def map_mwe(names=None, same_names=None, tags=None, pos=None):
    # make a map to go from ЖРИЦА ЛЮБВИ to {'жрица::любви_NOUN' : 'жрица_NOUN::любовь_NOUN'}
    my_map = defaultdict()

    for caps, tagged in zip(names, same_names):
        if ' ' in caps:
            old = preprocess_mwe(caps, tags=tags, pos=pos)
            new = tagged.replace(' ', '::')
            my_map[old.strip()] = new.strip()

    first_pairs = {k: my_map[k] for k in list(my_map)[:10]}
    print('First few matched items:', first_pairs, file=sys.stderr)

    return my_map


def new_preprocess_mwe(item, tags=None, pos=None, map_mwe_names=None):
    # Alas, those bigrams are overwhelmingly proper names while we need multi-word concepts.
    # For example, in aranea: "::[а-я]+\_PROPN" 8369 item, while the freq of all "::" 8407
    errors = 0
    new_item = None
    if pos == 'VERB':
        if len(item.split()) > 1:
            if tags:
                item = '::'.join(item.lower().split())
                new_item = item + '_VERB'
                # if map_mwe_names:
                #     try:
                #         new_item = map_mwe_names[item]
                #         # print(new_item)
                #     except KeyError:
                #         new_item = item
                #         errors += 1
                #         print("ERRRORRR", item)
                # else:
                #     new_item = item

            else:
                pass
            # print(item)

        else:
            if tags:
                item = item.lower()
                new_item = item + '_VERB'
            else:
                new_item = item.lower()

    elif pos == 'NOUN':
        if len(item.split()) > 1:
            if tags:
                item = '::'.join(item.lower().split())
                item = item + '_NOUN'
                # if item == 'часть::тело_NOUN':
                #     new_item = map_mwe_names[item.strip()]
                #     print('+++++++++', new_item) ##KeyError
                if map_mwe_names:
                    try:
                        new_item = map_mwe_names[item.strip()]
                    except KeyError:
                        new_item = item
                        errors += 1
                        # print("ERRRORRR", item)
                else:
                    new_item = item
            else:
                new_item = '::'.join(item.lower().split())

        else:

            if tags:
                item = item.lower()
                new_item = item + '_NOUN'
            else:
                new_item = item.lower()
    return new_item, errors


def preprocess_mwe(item, tags=None, pos=None):
    # Alas, those bigrams are overwhelmingly proper names while we need multi-word concepts.
    # For example, in aranea: "::[а-я]+\_PROPN" 8369 item, while the freq of all "::" 8407
    if pos == 'VERB':
        if len(item.split()) > 1:
            if tags:
                item = '::'.join(item.lower().split())
                item = item + '_VERB'

            else:
                item = '::'.join(item.lower().split())
            # print(item)

        else:
            if tags:
                item = item.lower()
                item = item + '_VERB'
            else:
                item = item.lower()

    elif pos == 'NOUN':
        if len(item.split()) > 1:
            if tags:
                item = '::'.join(item.lower().split())
                item = item + '_NOUN'
            else:
                item = '::'.join(item.lower().split())

        else:
            if tags:
                item = item.lower()
                item = item + '_NOUN'
            else:
                item = item.lower()

    return item


def convert_item_format(caps_word, tags=None, mwe=None, pos=None):
    if tags:
        if mwe:
            # this returns lowercased and tagged single words ot MWE
            item = preprocess_mwe(caps_word, tags=tags, pos=pos)
        else:
            if pos == 'VERB':
                item = caps_word.lower() + '_VERB'
            elif pos == 'NOUN':
                item = caps_word.lower() + '_NOUN'
            else:
                item = caps_word.lower()
    else:
        if mwe:
            item = preprocess_mwe(caps_word, tags=tags, pos=pos)  # this returns lowercased words
        else:
            item = caps_word.lower()

    return item


# now this function stores lowercased word pairs regardless of the combination
# of tags/mwe boolean options)
def preprocess_wordpair(pairs, tags=None, mwe=None, pos=None):
    preprocessed_train = []
    for hypo, hyper in pairs:
        if tags:
            if mwe:
                # this returns lowercased and tagged single words ot MWE
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                hyper = preprocess_mwe(hyper, tags=tags, pos=pos)
                preprocessed_train.append((hypo, hyper))
            else:
                if pos == 'VERB':
                    hypo = hypo.lower() + '_VERB'
                    hyper = hyper.lower() + '_VERB'
                    preprocessed_train.append((hypo, hyper))
                elif pos == 'NOUN':
                    hypo = hypo.lower() + '_NOUN'
                    hyper = hyper.lower() + '_NOUN'
                    preprocessed_train.append((hypo, hyper))

        # this is only when I can afford to retain all items with untagged fasttext
        else:
            if mwe:  # this returns lowercased words
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                hyper = preprocess_mwe(hyper, tags=tags, pos=pos)
                # preprocess_mwe returns lowercased items already
                preprocessed_train.append((hypo, hyper))
            else:
                preprocessed_train.append((hypo.lower(), hyper.lower()))

    return preprocessed_train


def preprocess_hypo(pairs, tags=None, mwe=None, pos=None):
    preprocessed_train = []
    for hypo, hyper in pairs:
        if tags:
            # this returns lowercased and tagged single words or MWE of type жрица::любовь_NOUN
            if mwe:
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                preprocessed_train.append((hypo, hyper))
            else:
                if pos == 'VERB':
                    hypo = hypo.lower() + '_VERB'
                    preprocessed_train.append((hypo, hyper))
                elif pos == 'NOUN':
                    hypo = hypo.lower() + '_NOUN'
                    preprocessed_train.append((hypo, hyper))
        # this is only when I can afford to retain all items with untagged fasttext
        else:
            if mwe:  # this returns lowercased words
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                preprocessed_train.append((hypo, hyper))
            else:
                preprocessed_train.append((hypo.lower(), hyper))

    return preprocessed_train


def write_hyp_pairs(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='unix', delimiter='\t', lineterminator='\n',
                            escapechar='\\', quoting=csv.QUOTE_NONE)
        writer.writerow(['hyponym', 'hypernym'])
        for pair in data:
            writer.writerow(pair)


def learn_projection(dataset, embedding, lmbd=1.0, from_df=False):
    print('Lambda: %.1f' % lmbd)
    if from_df:
        source_vectors = dataset['hyponym'].T
        target_vectors = dataset['hypernym'].T
    else:
        # gets two lists of vectors
        source_vectors = dataset[0]
        target_vectors = dataset[1]
    source_vectors = np.mat([[i for i in vec] for vec in source_vectors])
    target_vectors = np.mat([[i for i in vec] for vec in target_vectors])
    m = len(source_vectors)
    x = np.c_[np.ones(m), source_vectors]  # Adding bias term to the source vectors

    num_features = embedding.vector_size

    # Build initial zero transformation matrix
    learned_projection = np.zeros((num_features, x.shape[1]))
    learned_projection = np.mat(learned_projection)

    for component in range(0, num_features):  # Iterate over input components
        y = target_vectors[:, component]  # True answers
        # Computing optimal transformation vector for the current component
        cur_projection = normalequation(x, y, lmbd, num_features)

        # Adding the computed vector to the transformation matrix
        learned_projection[component, :] = cur_projection.T

    return learned_projection


def normalequation(data, target, lambda_value, vector_size):
    regularizer = 0
    if lambda_value != 0:  # Regularization term
        regularizer = np.eye(vector_size + 1)
        regularizer[0, 0] = 0
        regularizer = np.mat(regularizer)
    # Normal equation:
    theta = np.linalg.pinv(data.T * data + lambda_value * regularizer) * data.T * target
    return theta


def estimate_sims(source, targets, projection, model):
    #  Finding how far away are true targets from transformed sources
    test = np.mat(model[source])
    test = np.c_[1.0, test]  # Adding bias term
    predicted_vector = np.dot(projection, test.T)
    predicted_vector = np.squeeze(np.asarray(predicted_vector))
    target_vecs = [model[target] for target in targets]
    sims = [np.dot(unitvec(predicted_vector), unitvec(target_vec)) for target_vec in target_vecs]
    return sims


def predict(source, embedding, projection, topn=10):
    try:
        test = np.mat(embedding[source])
    except KeyError:
        return None

    test = np.c_[1.0, test]  # Adding bias term
    predicted_vector = np.dot(projection, test.T)
    predicted_vector = np.squeeze(np.asarray(predicted_vector))
    # Our predictions:
    nearest_neighbors = embedding.most_similar(positive=[predicted_vector], topn=topn)
    return nearest_neighbors, predicted_vector


def star_predict(source, embedding, projection, topn=10):
    try:
        test = np.mat(embedding[source])
    except KeyError:
        return None

    test = np.c_[1.0, test]  # Adding bias term
    predicted_vector = np.dot(projection, test.T)
    predicted_vector = np.squeeze(np.asarray(predicted_vector))
    # Our predictions:
    nearest_neighbors = embedding.most_similar(positive=[predicted_vector], topn=topn)
    return nearest_neighbors, predicted_vector


def popular_generic_concepts(relations_path):
    # how often an id has a name="hypernym" or "domain" when "child" in synset_relations.N.xml
    # (aim for the ratio hypernym/hyponym > 1)
    parsed_rels = read_xml(relations_path)
    freq_hypo = defaultdict(int)
    freq_hyper = defaultdict(int)
    for rel in parsed_rels:
        # in synset_relations the name of relations is assigned wrt the child,
        # it is the name of a child twds the parent, it seems
        identifier = rel.getAttributeNode('child_id').nodeValue
        name = rel.getAttributeNode('name').nodeValue
        if name == 'hypernym':  # or name == 'domain':
            freq_hyper[identifier] += 1
        elif name == 'hyponym':
            freq_hypo[identifier] += 1

    all_ids = list(freq_hypo.keys()) + list(freq_hyper.keys())

    ratios = defaultdict(int)
    for identifier in all_ids:
        try:
            ratios[identifier] = freq_hyper[identifier] / freq_hypo[identifier]
        except ZeroDivisionError:
            continue

    sort_it = {k: v for k, v in sorted(ratios.items(), key=lambda item: item[1], reverse=True)}

    my_ten = []
    for i, (k, v) in enumerate(sort_it.items()):
        if i < 10:
            my_ten.append(k)

    return my_ten  # synset ids


# parse ruwordnet and get a list of (synset_id, word) tuples for both one word and heads in MWE)
# mode 'main'
def filtered_dicts_mainwds_option(senses, tags=None, pos=None, mode=None, emb_voc=None,
                                  mapping=None):
    # this is where I want to include support for averaged MWE,
    doc = minidom.parse(senses)
    parsed_senses = doc.getElementsByTagName("sense")
    all_id_senses = []
    covered_ids = set()

    for sense in parsed_senses:

        identifier = sense.getAttributeNode('synset_id').nodeValue
        # lemma = sense.getAttributeNode("lemma").nodeValue ## changed from name to lemma
        name = sense.getAttributeNode("name").nodeValue  # changed back due to mismatch with the map
        main_wd = sense.getAttributeNode("main_word").nodeValue

        if len(name.split()) == 0:
            print('Missing name for a sense in synset %s' % identifier)
        # get MWE, singles compatible with embeddings already (lower, tagged)
        if mapping:

            item, _ = new_preprocess_mwe(name, tags=tags, pos=pos, map_mwe_names=mapping)
            # if name == 'ЧАСТЬ ТЕЛА':
            #     print('===========',item)
        else:
            item = preprocess_mwe(name, tags=tags, pos=pos)

        if mode == 'single':
            if item in emb_voc:
                all_id_senses.append((identifier, item))

        elif mode == 'main':
            if item in emb_voc:
                covered_ids.add(identifier)
                all_id_senses.append((identifier, item))

            if '::' in item:
                if identifier not in covered_ids:  # get the main word
                    if mapping:
                        item, _ = new_preprocess_mwe(main_wd, tags=tags, pos=pos,
                                                     map_mwe_names=mapping)
                    else:
                        item = preprocess_mwe(main_wd, tags=tags, pos=pos)
                    if item in emb_voc:
                        # only if the respective synset has not been covered already in the
                        # unconditional single word crawl
                        # activate if you want just one head word added from a non-singleword
                        # synset, not all of them (probably duplicates)
                        covered_ids.add(identifier)
                        all_id_senses.append((identifier, item))
            else:
                # print('What do you want to do with senses that are lexicalised as MWE?')
                continue
    nam2ids = defaultdict(list)
    for i in all_id_senses:
        synset = i[0].strip()
        name = i[1]
        nam2ids[name].append(synset)
    # print(lemmas2ids['часть_NOUN::тело_NOUN']) # empty list
    # reverse the dict to feed to synsets_vectorized, which takes id2lemmas
    id2names = defaultdict(list)
    for k, values in nam2ids.items():
        for v in values:
            id2names[v.strip()].append(k)

    # uncovered = uniq_ids - covered_ids
    # print('Uncoverd:', len(uncovered))
    # for i in uncovered:
    #     print(id2name[i])

    return nam2ids, id2names


# topn - how many similarities to retain from vector model to find the intersections
# with ruwordnet: less than 500 can return less than 10 candidates

# {'родитель_NOUN': ['147272-N', '136129-N', '5099-N', '2655-N']
def lemmas_based_hypers(test_item, vec=None, emb=None, topn=None, dict_w2ids=None,
                        limit=None):
    # enhanced cooc and hearst stats account for MWE matches,
    # but these can only be used if represented by embeddings!
    hyper_vec = np.array(vec, dtype=float)
    # default for mwe_vectors 100
    nearest_neighbors = emb.most_similar(positive=[hyper_vec], topn=topn)
    sims = []
    for res in nearest_neighbors:
        hypernym = res[0]
        similarity = res[1]
        if hypernym in dict_w2ids:
            # synset = dict_w2ids[hypernym][0] # limit the number if id to the first one only
            # we are adding as many tuples as there are synset ids associated with the topN
            # most_similar in embeddings and found in ruWordnet
            # and there is NO way to add matches from MWE unless they appear in the embeddings
            # in the current setup, when similarities are chosen from the default emb model
            # this dict is filtered through wordnet already; and it is here
            # where I add all ids of a hypernym
            for synset in dict_w2ids[hypernym]:
                if len(sims) < limit:
                    sims.append((synset, hypernym, similarity))
    # sort the list of tuples (id, sim) by the 2nd element and deduplicate
    # by rewriting the list while checking for duplicate synset ids
    # why do I do that?? they are already in the descending order by similarity??
    # sims = sorted(sims, key=itemgetter(2), reverse=True)

    # exclude hypernyms lemmas that match the query and lemmas from the same synset
    deduplicated_sims = []
    temp = set()
    nosamename = 0
    dup_ids = 0

    # sims_limited = sims[:limit]

    for a, b, c in sims:
        if test_item != b:
            if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b))  # (hypernym_synset_id, hypernym_wd)
            else:
                dup_ids += 1
                # print('Duplicate id among this items 100top similars', dup_ids)
        else:
            nosamename += 1
            # print('Query word = hypernym for this item: %s' % nosamename)

    return deduplicated_sims  # not limited to 10 now


def synsets_vectorized(emb=None, id2lemmas=None, named_synsets=None):
    total_lemmas = 0
    ruthes_oov = 0
    mean_synset_vecs = []
    synset_ids_names = []
    # 26312 in main for news_pos
    # 144031-N:[аутизм_NOUN, аутическое::мышление_NOUN] filtered thru emb already
    # and getting main words to represent synsets if desired
    this_mean_vec = None
    for identifier, wordlist in id2lemmas.items():
        # print('==', id, named_synsets[id], wordlist)
        current_vector_list = []
        for w in wordlist:
            total_lemmas += 1
            if w in emb.vocab:
                # print('++', w, emb[w])
                current_vector_list.append(emb[w])
                current_array = np.array(current_vector_list)
                # average column-wise, getting a new row=vector of size 300
                this_mean_vec = np.mean(current_array, axis=0)
            else:
                ruthes_oov += 1
                this_mean_vec = None

        # <class 'numpy.ndarray'> [ 0.031227    0.04932501  0.0154615   0.04967201
        mean_synset_vecs.append(this_mean_vec)
        synset_ids_names.append((identifier, named_synsets[identifier]))

    # synset_ids_names has (134530-N, КУНГУР)
    return synset_ids_names, mean_synset_vecs


def mean_synset_based_hypers(test_item, vec=None, syn_ids=None, syn_vecs=None):
    temp = set()
    deduplicated_sims = []
    nosamename = 0

    hyper_vec = np.array(vec, dtype=float)
    syn_vecs_arr = np.array(syn_vecs)

    sims = np.dot(hyper_vec, syn_vecs_arr.T)
    # sorting in the reverse descending order
    my_top_idx = (np.argsort(sims, axis=0)[::-1])[:50]
    # list of tuples (id, name) ex. (134530-N, КУНГУР)
    my_top_syn_ids_name = [syn_ids[hyper] for hyper in my_top_idx]

    my_top_syn_ids = [identifier[0] for identifier in my_top_syn_ids_name]
    my_top_syn_names = [identifier[1] for identifier in my_top_syn_ids_name]

    my_top_sims = [sims[ind] for ind in my_top_idx]  # actual similarity values

    for a, b, c in zip(my_top_syn_ids, my_top_syn_names, my_top_sims):
        # exclude same word as the name of the hypernym synset
        # (many of these names are strangely VERBs)
        if test_item != b:
            if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b, c))
        else:
            nosamename += 1

    this_hypo_res = deduplicated_sims  # list of (synset_id, hypernym_word, sim)

    return this_hypo_res  # list of (synset_id, hypernym_synset_name, sim)


# dict_w2ids = кунгур_NOUN:[['134530-N']], corpus_freqs = агностик_NOUN
# ["атеист_NOUN", "человек_NOUN", "религия_NOUN", ...]
def cooccurence_counts(test_item, deduplicated_res, corpus_freqs=None, thres_cooc=None,
                       thres_dedup=None):
    # print('This is before factoring in the cooccurence stats\n', deduplicated_res[:10])
    test_item = test_item.lower() + '_NOUN'
    # print('%s co-occured with\n%s' % (test_item, corpus_freqs[test_item]))

    new_list = []
    # cooc_dict has all items; hearst_dict does not
    try:
        # is this silently ignores items that are not in dict?
        if len(corpus_freqs[test_item]) != 0:
            # avoid rewriting the dict with cooc-predicted hypers
            for i in corpus_freqs[test_item][:thres_cooc]:  # [word_NOUN, word_NOUN, word_NOUN]
                i = i.strip()
                # <- list of [(id1_1,hypernym1_NOUN), (id1_2,hypernym1), (id2_1,hypernym2)]
                for tup in deduplicated_res[:thres_dedup]:
                    if i == tup[1]:
                        new_list.append(tup)
        else:
            # print('NOCOOCCURRENCE:', test_item)  ## травести, точмаш, прет-а-порте, стечкин
            new_list = deduplicated_res[:10]

    except KeyError:
        # print('NOCOOCCURRENCE:', test_item)
        new_list = deduplicated_res[:10]

    if len(new_list) < 10:
        for tup in deduplicated_res:
            if tup not in new_list:
                new_list.append(tup)

    # print('New order of hypernyms\n%s' % new_list[:10])
    return new_list  # list of (synset_id, hypernym_synset_name)


# for each test word FILTER1
def disambiguate_hyper_syn_ids(hypo, list_to_filter=None, emb=None, ft_model=None,
                               index_tuples=None, mean_syn_vectors=None, tags=None, pos=None):
    one_comp = 0
    over_n = 0
    lemma2id_vec_dict = defaultdict(list)
    lemma2id_dict = defaultdict(list)
    item = preprocess_mwe(hypo, tags=tags, pos=pos)

    if item in emb.vocab:
        hypo_vec = emb[item]  # for top-hyper all OOV are already taken care of
    else:
        # falling to FT representation
        if '_' in item:
            item = item[:-5]
        hypo_vec = ft_model[item]
        print('Alert if not ft-vector OOV-strategy!')

        # id-based lookup dict for mean synset vectors
    syn_vectors_dict = defaultdict()
    for (syn, name), vec in zip(index_tuples, mean_syn_vectors):
        syn_vectors_dict[syn] = vec  # values are synsets averaged vectors, stored as ndarrays

    for tup in list_to_filter:
        hyper_id = tup[0]
        hyper_lemma = tup[1]
        hyper_id_mean_vec = syn_vectors_dict[hyper_id]

        lemma2id_dict[hyper_lemma].append(hyper_id)  # from hyper_lemma to hyper_ids lists
        lemma2id_vec_dict[hyper_lemma].append(hyper_id_mean_vec)

    this_hypo_bestids = []
    for lemma, vecs in lemma2id_vec_dict.items():
        sims = [np.dot(hypo_vec, np.squeeze(np.asarray(vec)).T) for vec in vecs]
        my_top_idx = np.argmax(sims)
        if len(sims) == 1:
            one_comp += 1
        if len(sims) > 3:
            over_n += 1
        bestid = lemma2id_dict[lemma][int(my_top_idx)]
        this_hypo_bestids.append((bestid, lemma))
        # print(int(my_top_idx), len(lemma2id_dict[lemma]))

    tot = len(lemma2id_dict)

    return this_hypo_bestids, one_comp, over_n, tot  # list of (synset_id, hypernym_word) and stats


def get_generations(relations, redundant=None):
    parsed_rels = read_xml(relations)  # relations are defined wrt child
    rel_lookup = []
    for rel in parsed_rels:
        parent = rel.getAttributeNode('parent_id').nodeValue
        child = rel.getAttributeNode('child_id').nodeValue
        rel_name = rel.getAttributeNode('name').nodeValue
        if redundant == 'kid':
            if rel_name == 'hyponym':
                rel_lookup.append((child, parent))  # first order parents
        elif redundant == 'parent':
            if rel_name == 'hypernym':
                rel_lookup.append((child, parent))  # reverse order of rel i.e child==parent
        else:
            rel_lookup = None
            print("Do you want to lose kids or parents?")

    print('All child-parent', len(rel_lookup))  # including parent-grandparent

    return rel_lookup  # , rel_lookup2


def lose_family_anno(hypo, deduplicated_res, rel_lookup):
    # get a list of ids from the list of tuples (id, hyper_word)
    predicted_ids = [i[0] for i in deduplicated_res]
    combos = []  # get all combinations of predicted ids
    for i in itertools.combinations(predicted_ids, 2):
        combos.append(i)

    # find combinations where one el is a child of the other and delete this child
    hits = []
    for combo in combos:
        # PROBLEM: retain dog if all three are in the output
        # preds for poodle: collie - dog - animal,
        # but this assumes the relatedness of predicted synsets
        if combo in rel_lookup:
            for (identifier, word) in deduplicated_res:
                if identifier == combo[0]:
                    print('Pruned child:', hypo, (word, combo[0]))
                    hits.append((identifier, word))

    this_hypo_res = [x for x in deduplicated_res if x not in hits]

    print('Pruned kids:', len(hits))
    print('==Smaller? %d -> %d' % (len(deduplicated_res), len(set(this_hypo_res))))

    return this_hypo_res


def lose_family_comp(hypo, deduplicated_res, train=None, redundant=None):
    this_hypo_res = []

    # build a graph, get the list of connected components
    # (hm, not exaustive! does not account for same-id components)
    components, synset2word, _, synset2parents = get_data(train)

    # get a list of ids from the list of tuples (id, hyper_word)
    predicted_ids = [i[0] for i in deduplicated_res]
    combos = []  # get all combinations of predicted ids
    for i in itertools.combinations(predicted_ids, 2):
        combos.append(i)

    hits = []
    # iterate over all components (the list of components is not exaustive it seems,
    # but parents include all parents of hypo synset)
    # in effect it is iteration over hyponym ids!
    # all hyponym ids that have associated lists of hypernyms corresponding to
    # lines in the training data
    for i in range(len(components)):
        # if i == 1:
        #     print(components[i])
        out = []
        parents_ids = []
        for identifier in components[i]:  # iterate over synsets in this component
            # синсеты-родители синсетов-гипонимов (с учетом второго порядка) в этом компоненте
            for parents in synset2parents[identifier]:
                parents_ids.extend(parents)
        # find combinations where one el is a child of the other in this component
        # and delete this child
        for combo in combos:
            if combo[0] in components[i] and combo[1] in parents_ids:
                for (identifier, word) in deduplicated_res:
                    if redundant == 'kid':
                        if identifier == combo[0]:
                            out.append(identifier)
                            print('Prune child:', hypo, (word, combo))
                    elif redundant == 'parent':
                        if identifier == combo[1]:
                            out.append(identifier)
                            print('Prune parent:', hypo, (word, combo))
                    else:
                        out = None
                        print("Do you want to lose kids or parents?")
        # deleting by re-writing the list
        for res in deduplicated_res:
            for identifier in out:
                if identifier != res[0]:
                    this_hypo_res.append(res)
                else:
                    hits.append(res)

    print('Pruned kids:', len(hits))
    print('==Smaller? %d -> %d' % (len(deduplicated_res), len(set(this_hypo_res))))

    return this_hypo_res


######################
if __name__ == '__main__':
    print('=== This is a modules script, it is not supposed to run as main ===')
