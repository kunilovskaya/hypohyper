import csv
import os, sys, re
from xml.dom import minidom
from collections import defaultdict
from collections import Counter
import pandas as pd
import logging
from gensim.models import KeyedVectors
import numpy as np
from gensim.matutils import unitvec
from smart_open import open
from itertools import repeat
from operator import itemgetter
import json


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


def wd2id_dict(id2dict): # input: id2wds
    wd2ids = defaultdict(list)
    for k, values in id2dict.items():
        for v in values:
            wd2ids[v].append(k)

    return wd2ids  # ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]


# FYI: distribution of relations annotated in ruwordnet
def get_all_rels(relations): # <- parsed_rels = read_xml(args.relations) <- synset_relations.N.xml
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

    my_TEXTS = df_train['TEXT'].tolist()
    my_PARENT_TEXTS = df_train['PARENT_TEXTS'].tolist()
    
    all_pairs = []
    
    for hypo, hyper in zip(my_TEXTS, my_PARENT_TEXTS):
        hypo = hypo.replace(r'"', '')
        hyper = hyper.replace(r'"', '')
        hypo = hypo.split(', ')
        hyper = hyper.split(', ')
        
        for i in hypo:
            wd_tuples = list(zip(repeat(i), hyper))
            all_pairs.append(wd_tuples)
    all_pairs = [item for sublist in all_pairs for item in sublist]  # flatten the list
    
    return all_pairs # ('ИХТИОЛОГ', 'УЧЕНЫЙ')


def process_tsv_deworded_hypers(filepath):
    ## open the original training data (or part of it) with json
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
            temp_dict[w] = json.loads(par_ids)  # {'WORD': ['4544-N', '147272-N'], '120440-N': ['141697-N', '116284-N']}
    
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
        par_ids = par_ids.replace("'", '"')  ## to meet the json requirements
        id_list = json.loads(par_ids)
        for id in id_list:
            temp_dict[wd].append(id)  # {'WORD': [['4544-N'], ['147272-N']], 'WORD': ['141697-N', '116284-N']}

    for hypo_w, hypers_ids in temp_dict.items():
        id_tuples = list(zip(repeat(hypo_w), hypers_ids))
        org_pairs.append(id_tuples)

    org_pairs = [item for sublist in org_pairs for item in sublist]  # flatten the list
    print('Number of wd-to-synset pairs in training_data: ', len(org_pairs))
    
    return org_pairs # ('ИХТИОЛОГ', '9033-N')


def get_orgtrain(filepath, map=None): # map = synset_words
    lines = open(filepath, 'r').readlines()
    temp_dict = defaultdict(list)
    org_pairs = []
    for line in lines:
        res = line.split('\t')
        wd, par_ids = res
        wd = wd.replace(r'"', '')  # get rid of dangerous quotes in МАШИНА "ЖИГУЛИ"
        par_ids = par_ids.replace("'", '"')  # to meet the json requirements
        id_list = json.loads(par_ids)
        for id in id_list:
            wd_list = map[id]
            for i in wd_list:
                temp_dict[wd].append(i)
    for hypo_w, hypers_wds in temp_dict.items():
        wd_tuples = list(zip(repeat(hypo_w), hypers_wds))
        org_pairs.append(wd_tuples)
    org_pairs = [item for sublist in org_pairs for item in sublist]  # flatten the list
    print('Number of wd-to-synset pairs in training_data: ', len(org_pairs))
    
    return org_pairs  ## ('ИХТИОЛОГ', '9033-N')

def get_orgtest_deworded(filepath):
    mwe = []
    lines = open(filepath, 'r').readlines()
    gold_dict = defaultdict(list)
    for line in lines:
        res = line.split('\t')
        wd, par_ids = res
        wd = wd.replace(r'"', '') # get rid of dangerous quotes in МАШИНА "ЖИГУЛИ"
        
        if len(wd.split()) != 1 or bool(re.search('[a-zA-Z]', wd)): ## skip MWE
            mwe.append(wd)
            continue
        par_ids = par_ids.replace("'", '"') # to meet the json requirements
        id_list = json.loads(par_ids)

        gold_dict[wd].append(id_list)  # {'WORD1': [['4544-N'], ['147272-N']], 'WORD2': [['141697-N', '116284-N']]}
    
    return gold_dict


def read_train(tsv_in):
    df_out = pd.read_csv(tsv_in, sep='\t')
    return df_out


def load_embeddings(modelfile):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if not os.path.isfile(modelfile):
        raise FileNotFoundError("No file called {file}".format(file=modelfile))
    # Determine the model format by the file extension
    if modelfile.endswith('.bin.gz') or modelfile.endswith('.bin') or modelfile.endswith('.w2v'):  # Binary word2vec file
        emb_model = KeyedVectors.load_word2vec_format(modelfile, binary=True,
                                                      unicode_errors='replace', limit=3500000)
    elif modelfile.endswith('.txt.gz') or modelfile.endswith('.txt') \
            or modelfile.endswith('.vec.gz') or modelfile.endswith('.vec'):  # Text word2vec file
        emb_model = KeyedVectors.load_word2vec_format(modelfile, binary=False,
                                                      unicode_errors='replace', limit=3500000)
    else:  # Native Gensim format, inclufing for fasttext models (.model in a folder with the other support files)
        emb_model = KeyedVectors.load(modelfile)
    emb_model.init_sims(replace=True)

    return emb_model

# def get_vector(word, emb=None):
#     if not emb:
#         return None
#     vector = emb[word]
#     return vector

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
            item = preprocess_mwe(caps_word, tags=tags, pos=pos) # this returns lowercased and tagged single words ot MWE
        else:
            if pos == 'VERB':
                item = caps_word.lower() + '_VERB'
            elif pos == 'NOUN':
                item = caps_word.lower() + '_NOUN'
            else:
                item = caps_word.lower()
    else:
        if mwe:
            item = preprocess_mwe(caps_word, tags=tags, pos=pos) # this returns lowercased words
        else:
            item = caps_word.lower()
            
    return item
    
## now this function stores lowercased word pairs regardless of the combination of tags/mwe boolean options)
def preprocess_wordpair(pairs, tags=None, mwe=None, pos=None):
    preprocessed_train = []
    for hypo, hyper in pairs:
        if tags:
            if mwe:
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos) # this returns lowercased and tagged single words ot MWE
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
                            
        ## this is only when I can afford to retain all items with untagged fasttext
        else:
            if mwe: ## this returns lowercased words
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                hyper = preprocess_mwe(hyper, tags=tags, pos=pos)
                preprocessed_train.append((hypo, hyper)) ## preprocess_mwe returns lowercased items already
            else:
                preprocessed_train.append((hypo.lower(), hyper.lower()))

    return preprocessed_train


def preprocess_hypo(pairs, tags=None, mwe=None, pos=None):
    preprocessed_train = []
    for hypo, hyper in pairs:
        if tags:
            if mwe:  # this returns lowercased and tagged single words ot MWE
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                preprocessed_train.append((hypo, hyper))
            else:
                if pos == 'VERB':
                    hypo = hypo.lower() + '_VERB'
                    preprocessed_train.append((hypo, hyper))
                elif pos == 'NOUN':
                    hypo = hypo.lower() + '_NOUN'
                    preprocessed_train.append((hypo, hyper))
        ## this is only when I can afford to retain all items with untagged fasttext
        else:
            if mwe:  # this returns lowercased words
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                preprocessed_train.append((hypo, hyper))
            else:
                preprocessed_train.append((hypo.lower(), hyper))
    
    return preprocessed_train

def write_hyp_pairs(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='unix', delimiter='\t', lineterminator='\n', escapechar='\\', quoting=csv.QUOTE_NONE)
        writer.writerow(['hyponym', 'hypernym'])
        for pair in data:
            writer.writerow(pair)


def learn_projection(dataset, embedding, lmbd=1.0, from_df=False):
    print('Lambda: %d' % lmbd)
    if from_df:
        source_vectors = dataset['hyponym'].T
        target_vectors = dataset['hypernym'].T
    else:
        ## gets two lists of vectors
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
    ## how often an id has a name="hypernym" when "parent" in synset_relations.N.xml (aim for the ratio hypernym/hyponym > 1)
    parsed_rels = read_xml(relations_path)
    freq_hypo = defaultdict(int)
    freq_hyper = defaultdict(int)
    for rel in parsed_rels:
        ## in WordNet relations the name of relations is assigned wrt the child, it is the name of a child twds the parent, it seems
        id = rel.getAttributeNode('child_id').nodeValue
        name = rel.getAttributeNode('name').nodeValue
        if name == 'hypernym':
            freq_hyper[id] += 1
        elif name == 'hyponym':
            freq_hypo[id] += 1
    
    all_ids = list(freq_hypo.keys()) + list(freq_hyper.keys())
    
    ratios = defaultdict(int)
    for id in all_ids:
        try:
            ratios[id] = freq_hyper[id] / freq_hypo[id]
        except ZeroDivisionError:
            continue
    
    sort_it = {k: v for k, v in sorted(ratios.items(), key=lambda item: item[1], reverse=True)}

    my_ten = []
    for i, (k, v) in enumerate(sort_it.items()):
        if i < 10:
            my_ten.append(k)
    
    return my_ten # synset ids


####### parse ruwordnet and get a list of (synset_id, word) tuples for both one word and heads in MWE)
def filtered_dicts_mainwds_option(senses, tags=None, pos=None, mode=None, emb_voc=None):  # mode 'main'
    
    doc = minidom.parse(senses)
    parsed_senses = doc.getElementsByTagName("sense")
    all_id_senses = []
    covered_ids = set()

    for sense in parsed_senses:
        
        id = sense.getAttributeNode('synset_id').nodeValue
        name = sense.getAttributeNode("name").nodeValue
        main_wd = sense.getAttributeNode("main_word").nodeValue
        if len(name.split()) == 0:
            item = None
            print('Missing name for a sense in synset %s' % id)
        
        item = preprocess_mwe(name, tags=tags,
                              pos=pos)  # get MWE, singles compatible with embeddings already (lower, tagged)
        
        if '::' not in item:
            if item in emb_voc:
                covered_ids.add(id)
                all_id_senses.append((id, item))
            else:
                continue
        
        elif '::' in item and mode == 'main':
            if item in emb_voc:
                ## adding the few cases of MWE in embeddings vocabulary
                all_id_senses.append((id, item))
            else:
                if id not in covered_ids:
                    item = preprocess_mwe(main_wd, tags=tags, pos=pos)
                    ## only if the respective synset has not been covered already in the unconditional single word crawl
                    covered_ids.add(
                        id)  # activate if you want just one head word added from a non-singleword synset, not all of them (probably duplicates)
                    all_id_senses.append((id, item))
        else:
            # print('What do you want to do with senses that are lexicalised as MWE?')
            continue
            

    lemmas2ids = defaultdict(list)
    for i in all_id_senses:
        synset = i[0]
        name = i[1]
        lemmas2ids[name].append(synset)

    ## reverse the dict to feed to synsets_vectorized, which takes id2lemmas
    id2lemmas = defaultdict(list)
    for k, values in lemmas2ids.items():
        for v in values:
            id2lemmas[v].append(k)
      
    # all synsets can be represented with embeddings now
    return lemmas2ids, id2lemmas

# topn - how many similarities to retain from vector model to find the intersections with ruwordnet: less than 500 can return less than 10 candidates

def lemmas_based_hypers(test_item, vec=None, emb=None, ft_model=None, topn=None, dict_w2ids=None, index_tuples=None, mean_syn_vectors=None, filt1=None): # {'родитель_NOUN': ['147272-N', '136129-N', '5099-N', '2655-N']
    
    hyper_vec = np.array(vec, dtype=float)
    nearest_neighbors = emb.most_similar(positive=[hyper_vec], topn=topn) # words
    sims = []
    for res in nearest_neighbors:
        hypernym = res[0]
        similarity = res[1]
        if hypernym in dict_w2ids:
            ## we are adding as mane tuples as there are synset ids associated with the topN most_similar in embeddings and found in ruWordnet
            for synset in dict_w2ids[hypernym]:
                sims.append((synset, hypernym, similarity))
    # sort the list of tuples (id, sim) by the 2nd element and deduplicate
    # by rewriting the list while checking for duplicate synset ids
    sims = sorted(sims, key=itemgetter(2), reverse=True)
    ## exclude hypernyms lemmas that match the query and lemmas from the same synset
    deduplicated_sims = []
    temp = set()
    nosamename = 0
    dup_ids = 0
    
    sims100 = sims[:100]
    
    for a, b, c in sims100:
        if test_item != b:
           if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b))
           else:
               dup_ids += 1
               # print('Duplicate id among this items 100top similars', dup_ids)
        else:
            nosamename += 1
            # print('Query word = hypernym for this item: %s' % nosamename)

    return deduplicated_sims ## not limited to 10

def synsets_vectorized(emb=None, id2lemmas=None, named_synsets=None, tags=None, pos=None):
    total_lemmas = 0
    ruthes_oov = 0
    mean_synset_vecs = []
    synset_ids_names = []
    for id, wordlist in id2lemmas.items(): # 144031-N:[аутизм_NOUN, аутическое::мышление_NOUN] filtered thru emb already and getting main words to represent synsets if desired
        # print('==', id, named_synsets[id], wordlist)
        current_vector_list = []
        for w in wordlist:
            total_lemmas += 1
            if w in emb.vocab:
                # print('++', w, emb[w])
                current_vector_list.append(emb[w])
                current_array = np.array(current_vector_list)
                this_mean_vec = np.mean(current_array,axis=0)  # average column-wise, getting a new row=vector of size 300
                mean_synset_vecs.append(this_mean_vec) # <class 'numpy.ndarray'> [ 0.031227    0.04932501  0.0154615   0.04967201
                synset_ids_names.append((id, named_synsets[id]))
            else:
                ruthes_oov += 1
                this_mean_vec = None

    ## synset_ids_names has (134530-N, КУНГУР)
    return synset_ids_names, mean_synset_vecs

def mean_synset_based_hypers(test_item, vec=None, syn_ids=None, syn_vecs=None, topn=10):
    sims = []
    temp = set()
    deduplicated_sims = []
    nosamename = 0
    
    hyper_vec = np.array(vec, dtype=float)
    syn_vecs_arr = np.array(syn_vecs)
    
    sims = np.dot(hyper_vec, syn_vecs_arr.T)
    ## sorting in the reverse descending order
    my_top_idx = (np.argsort(sims, axis=0)[::-1])[:50]
    
    my_top_syn_ids_name = [syn_ids[hyper] for hyper in my_top_idx] ## list of tuples (id, name) ex. (134530-N, КУНГУР)
    
    my_top_syn_ids = [id[0] for id in my_top_syn_ids_name]
    my_top_syn_names = [id[1] for id in my_top_syn_ids_name]
    
    my_top_sims = [sims[ind] for ind in my_top_idx] ##actual similarity values

    for a, b, c in zip(my_top_syn_ids, my_top_syn_names, my_top_sims):
        # exclude same word as the name of the hypernym synset (many of these names are strangely VERBs)
        if test_item != b:
            if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b, c))
        else:
            nosamename += 1

    this_hypo_res = deduplicated_sims[:topn]  # list of (synset_id, hypernym_word, sim)
    
    return this_hypo_res  # list of (synset_id, hypernym_synset_name, sim)

## dict_w2ids = кунгур_NOUN:[['134530-N']], corpus_freqs = агностик_NOUN ["атеист_NOUN", "человек_NOUN", "религия_NOUN", ...]
def cooccurence_counts(test_item, vec=None, emb=None, topn=None, dict_w2ids=None, corpus_freqs=None, method=None):
    nosamename = 0
    hyper_vec = np.array(vec, dtype=float)
    temp = set()
    deduplicated_sims = []
    nearest_neighbors = emb.most_similar(positive=[hyper_vec], topn=topn)
    sims = []
    for res in nearest_neighbors:
        hypernym = res[0] ## word_NOUN
        similarity = res[1]
        if hypernym in dict_w2ids:
            for synset in dict_w2ids[hypernym]:  # we are adding all synset ids associated with the topN most_similar in embeddings and found in ruWordnet
                sims.append((synset, hypernym, similarity))
    
    # sort the list of tuples (id, sim) by the 2nd element and deduplicate
    # by rewriting the list while checking for duplicate synset ids
    sims = sorted(sims, key=itemgetter(2), reverse=True) ## those that are found in ruWordNet in top 500 most_similar embeddings
    
    for a, b, c in sims:
        # exclude same word as hypernym even if attributed to another synset
        b = b[:-5].upper()
        if test_item != b:
            # print(hypo, b)
            if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b, c))
        else:
            nosamename += 1
    # print('Selves as hypernyms: %s' % nosamename)
    
    this_hypo_res = deduplicated_sims[:10]  # list of (synset_id, HYPERNYM_WORD, sim)
    
    print('\n\n%%%%%%%%%%%%%%%%%%')
    print('MY QUERY:  === %s ===' % test_item)
    
    print('This is before factoring in the cooccurence stats\n', this_hypo_res)
    test_item = test_item.lower()+'_NOUN'
    # print('%s co-occured with\n%s' % (test_item, corpus_freqs[test_item]))
    
    limit_mobility = int(''.join([i for i in method if i.isdigit()]))
    new_list = []
    if len(corpus_freqs[test_item]) != 0:

        for i in corpus_freqs[test_item]: # [word_NOUN, word_NOUN, word_NOUN]
            for tup in deduplicated_sims[:limit_mobility]:  ## maybe further limit these 500 words to 100?
                if i == tup[1].lower()+'_NOUN':
                    # print('==', i, tup[1].lower()+'_NOUN')
            # if i in hypo_small: ## this is always the casebecause co-occurence candidates were drawn from the top (how many?) hypernyms
                    new_list.append(tup)
    else:
        # print('NOCOOCCURRENCE:', test_item) ## травести, точмаш, прет-а-порте, стечкин
        new_list = deduplicated_sims[:10]
    if len(new_list) < 10:
        for tup in deduplicated_sims:
            if tup not in new_list:
                new_list.append(tup)
                
    new_list = new_list[:10]
    
    print('New order of hypernyms\n%s' % (new_list))
    return new_list  # list of (synset_id, hypernym_synset_name)


# for each test word FILTER1
def disambiguate_hyper_syn_ids(hypo, list_to_filter=None, emb=None, ft_model=None, index_tuples=None, mean_syn_vectors=None, tags=None, pos=None):
    one_comp = 0
    over_n = 0
    lemma2id_vec_dict = defaultdict(list)
    lemma2id_dict = defaultdict(list)
    item = preprocess_mwe(hypo, tags=tags, pos=pos)
    
    if item in emb.vocab:
        hypo_vec = emb[item]  ## for top-hyper all OOV are already taken care of
    else:
        ## falling to FT representation
        if '_' in item:
            item = item[:-5]
        hypo_vec = ft_model[item]
        print('Alert if not ft-vector OOV-strategy!')
        
        ## id-based lookup dict for mean synset vectors
    syn_vectors_dict = defaultdict()
    for (syn, name), vec in zip(index_tuples, mean_syn_vectors):
        syn_vectors_dict[syn] = vec  ## values are synsets averaged vectors, stored as ndarrays
    
    for tup in list_to_filter:
        hyper_id = tup[0]
        hyper_lemma = tup[1]
        hyper_id_mean_vec = syn_vectors_dict[hyper_id]
        
        lemma2id_dict[hyper_lemma].append(hyper_id)  ## from hyper_lemma to hyper_ids lists
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


######################
if __name__ == '__main__':
    print('=== This is a modules script, it is not supposed to run as main ===')
