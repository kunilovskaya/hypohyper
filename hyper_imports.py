import csv
import os, re
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
from scipy.spatial import distance
import scipy.spatial as sp
import json


# parse ruwordnet
# and (1) build one-to-one dicts for synsets_ids and any asociated lemmas/=words/=senses/=synonyms
def read_xml(xml_file):
    doc = minidom.parse(xml_file)
    # node = doc.documentElement
    try:
        # it is a list?
        parsed = doc.getElementsByTagName("synset") or doc.getElementsByTagName("relation")
    except TypeError:
        # nodelist = parsed
        print('Are you sure you are passing the expected files?')
        parsed = None

    return parsed  # a list of xml entities


# how does this account for the fact that wd can be a member of several synsets?:
# synset2wordlist mappings are unique, while the reverse is not true
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


def wd2id_dict(id2dict): # id2wds
    wd2id = defaultdict(list)
    for k, values in id2dict.items():
        for v in values:
            wd2id[v].append(k)

    # ex. ЗНАК:[152660-N, 118639-N, 107519-N, 154560-N]
    return wd2id


# distribution of relations annotated in ruwordnet
def get_all_rels(relations):
    my_list = []
    for rel in relations:
        rel_name = rel.getAttributeNode('name').nodeValue
        my_list.append(rel_name)
    my_dict = Counter(my_list)  # Counter({'apple': 3, 'egg': 2, 'banana': 1})

    # iterator = iter(my_dict.items())
    # for i in range(len(my_dict)):
    #     print(next(iterator))
    print('=====\n %s \n=====' % my_dict)


# (2) по лемме (ключи в нашем словаре) его синсет-гипероним и синсет-гипоним
# как списки лемм и номера соответствующих синсетов
# if you pass hyponym you get the query hyperonyms (ids)
# level: hypernym, hyponym, domain, POS-synonymy, instance hypernym, instance hyponym:
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


# level: hypernym, hyponym, domain, POS-synonymy, instance hypernym, instance hyponym:
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


def process_tsv(filepath, emb=None, tags=None, mwe=None, pos=None, skip_oov=None):
    df_train = read_train(filepath)
    
    # strip of [] and ' in the strings:
    ## TODO maybe average vectors for representatives of each synset in the training_data
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
    print('=== Raw training set: %s ===' % len(all_pairs))
    print('Raw examples:\n', all_pairs[:3])
    
    # limit training_data to the pairs that are found in the embeddings
    filtered_pairs = filter_dataset(all_pairs, emb, tags=tags, mwe=mwe, pos=pos, skip_oov=skip_oov)
    print('\n=== Embeddings coverage (intrinsuc train): %s ===' % len(filtered_pairs))
    
    # print('!!! WYSIWYG as lookup queries!!!')
    # print('Expecting: TAGS=%s; MWE=%s; %s' % (tags, mwe, pos))
    print(filtered_pairs[:3])
    mwes = [(a, b) for (a, b) in filtered_pairs if re.search('::', a) or re.search('::', b)]
    print(mwes[:3])
    # print('Number of MWE included %s' % len(mwes))
    
    return filtered_pairs ## ('ихтиолог_NOUN', 'ученый_NOUN')


def process_tsv_deworded(filepath, emb=None, tags=None, mwe=None, pos=None, skip_oov=None):
    ## open with json
    lines = open(filepath, 'r').readlines()
    temp_dict = {}
    synset_pairs = []
    for i,line in enumerate(lines):
        if i == 0:
            continue
        res = line.split('\t')
        ## learn the hypoWORD to averaged synset vector projection
        hypo_id, wds, par_ids, _ = res
        par_ids = par_ids.replace("'", '"')
        
        wds = wds.split(', ')
        for w in wds:
            w = w.replace(r'"', '')
            temp_dict[w] = json.loads(par_ids)  # {'WORD': ['4544-N', '147272-N'], '120440-N': ['141697-N', '116284-N']}
        ## alternatively use averaged vectors for hypo_ids (only 19327 training pairs and a problem with mapping input words to their synsets)
        ## fix quotes
        # temp_dict[hypo_id] = json.loads(par_ids) # {'126551-N': ['4544-N', '147272-N'], '120440-N': ['141697-N', '116284-N']}
    # for hypo_id, hypers_ids in temp_dict.items():
    #     id_tuples = list(zip(repeat(hypo_id), hypers_ids))
    #     synset_pairs.append(id_tuples)
        
    for hypo_w, hypers_ids in temp_dict.items():
        id_tuples = list(zip(repeat(hypo_w), hypers_ids))
        synset_pairs.append(id_tuples)
    
    synset_pairs = [item for sublist in synset_pairs for item in sublist]  # flatten the list
    print('Number of wd-to-synset pairs in training_data: ', len(synset_pairs))
    
    return synset_pairs ## ('ИХТИОЛОГ', 'УЧЕНЫЙ')

def process_test_tsv(filepath, emb=None, tags=None, mwe=None, pos=None, skip_oov=None):
    
    df = read_train(filepath)

    # strip of [] and ' in the strings,  json fails here with json.decoder.JSONDecodeError: Expecting value: line 1 column 2 (char 1)
    df = df.replace(to_replace=r"[\[\]']", value='', regex=True)
    
    my_TEXTS = df['TEXT'].tolist()
    my_PARENT_IDS = df['PARENTS'].tolist()
    
    good_pairs = []
    oov_counter = 0
    for hypos, hypers in zip(my_TEXTS, my_PARENT_IDS):
        hypos = hypos.split(', ')
        hypers = hypers.split(', ')
        
        for i in hypos:
            ## filter out multiwords
            # print(i)
            if len(i.split()) == 1:
                if tags:
                    if pos == 'NOUN':
                        i = i.lower() + '_NOUN'
                    elif pos == 'VERB':
                        i = i.lower() + '_VERB'
                else:
                    i = i
                ## filter out single-word OOV in the embeddings
                if i in emb.vocab:
                    wd_tuples = list(zip(repeat(i), hypers))
                    good_pairs.append(wd_tuples)
                else:
                    oov_counter += 1
                    # print('====',i)
            else:
                continue
    good_pairs = [item for sublist in good_pairs for item in sublist]  # flatten the list
    print('+++ Raw testset pairs: %s +++' % (len(good_pairs)+oov_counter))
    print('Examples from saved intrinsic test:\n', good_pairs[:3])
    print('\n=== Embeddings coverage (intrinsuc test): %.2f%% ===' % (100 - (oov_counter/(len(good_pairs))*100)))
    
    return good_pairs  ## ('ихтиолог_NOUN', 'ученый_NOUN')


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
    # print('Success! Vectors loaded')

    return emb_model


def get_vector(word, emb=None):
    if not emb:
        return None
    vector = emb[word]
    return vector


def preprocess_mwe(item, tags=None, pos=None):
    # Alas, those bigrams are overwhelmingly proper names while we need multi-word concepts.
    # For example, in aranea: "::[а-я]+\_NOUN" 8369 item, while the freq of all "::" 8407
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
            # print('MWE example untagged:', item)
        else:
            if tags:
                item = item.lower()
                item = item + '_NOUN'
            else:
                item = item.lower()
            
    return item

## now this function stores lowercased word pairs regardless of the combination of tags/mwe boolean options)
def filter_dataset(pairs, embedding, tags=None, mwe=None, pos=None, skip_oov=None):
    smaller_train = []
    for hypo, hyper in pairs:
        if tags:
            if mwe: ## this returns lowercased and tagged single words ot MWE
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                hyper = preprocess_mwe(hyper, tags=tags, pos=pos)
                if hypo in embedding.vocab and hyper in embedding.vocab:
                    smaller_train.append((hypo, hyper))
            else:
                if pos == 'VERB':
                    hypo = hypo.lower() + '_VERB'
                    hyper = hyper.lower() + '_VERB'
                    if hypo in embedding.vocab and hyper in embedding.vocab:
                        smaller_train.append((hypo, hyper))
                elif pos == 'NOUN':
                    hypo = hypo.lower() + '_NOUN'
                    hyper = hyper.lower() + '_NOUN'
                    if skip_oov == False:
                        smaller_train.append((hypo, hyper))
                    elif skip_oov == True:
                        if hypo in embedding.vocab and hyper in embedding.vocab:
                            smaller_train.append((hypo, hyper))
                            
        ## this is only when I can afford to retain all items with untagged fasttext
        else:
            if mwe: ## this returns lowercased words
                hypo = preprocess_mwe(hypo, tags=tags, pos=pos)
                hyper = preprocess_mwe(hyper, tags=tags, pos=pos)
                if skip_oov == False:
                    smaller_train.append((hypo, hyper))
                elif skip_oov == True:
                    if hypo in embedding.vocab and hyper in embedding.vocab:
                        smaller_train.append((hypo, hyper)) ## preprocess_mwe returns lowercased items already
                else:
                    smaller_train.append((hypo.lower(), hyper.lower()))
            else:
                ## this is tuned for ft vectors to filter out OOV (mostly MWE)
                if skip_oov == False:
                    smaller_train.append((hypo.lower(), hyper.lower()))
                elif skip_oov == True:
                    if hypo.lower() in embedding.vocab and hyper.lower() in embedding.vocab:
                        smaller_train.append((hypo.lower(), hyper.lower()))
                else:
                    smaller_train.append((hypo.lower(), hyper.lower()))

    return smaller_train  # only the pairs that are found in the embeddings if skip_oov=True



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
        ## this gets a tuple of two lists of vectors: (source_vecs, target_vecs)
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
    ## what happens when your test word is not in the embeddings? how do you get its vector?
    ## skip for now!
    ## TODO implement this
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
    ## what happens when your test word is not in the embeddings? how do you get its vector?
    ## skip for now!
    ## TODO implement this
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
    print(all_ids[:5])
    # print(len(set(all_ids)))
    
    ratios = defaultdict(int)
    for id in all_ids:
        try:
            ratios[id] = freq_hyper[id] / freq_hypo[id]
        except ZeroDivisionError:
            continue
    
    sort_it = {k: v for k, v in sorted(ratios.items(), key=lambda item: item[1], reverse=True)}
    # for id in sort_it:
    #     print(id, sort_it[id])
    my_ten = []
    for i, (k, v) in enumerate(sort_it.items()):
        if i < 10:
            my_ten.append(k)
            # print(k)
    
    
    return my_ten # synset ids


####### parse ruwordnet and get a list of (synset_id, word) tuples for both one word and heads in MWE)

def parse_taxonymy(senses, tags=None, pos=None, mode=None, emb_voc=None):

    doc = minidom.parse(senses)
    parsed_senses = doc.getElementsByTagName("sense")
    all_id_senses = []

    print('Total number of senses %d' % len(parsed_senses))
    count_main = 0
    ids = []
    for sense in parsed_senses:
        
        id = sense.getAttributeNode('synset_id').nodeValue
        name = sense.getAttributeNode("name").nodeValue
        main_wd = sense.getAttributeNode("main_word").nodeValue
        
        ids.append(id)
        if len(name.split()) == 0:
            item = None
            print('Missing name for a sense in synset %s' % id)
        
        if mode == 'single':
            if len(name.split()) == 1:
                if tags == True:
                    if pos == 'NOUN':
                        item = name.lower() + '_NOUN'
                    elif pos == 'VERB':
                        item = name.lower() + '_VERB'
                    else:
                        item = None
                        print('Which PoS part of WordNet are we dealing with?')
                else:
                    item = name.lower()
                
                all_id_senses.append((id, item))
            else:
                continue
        
        ## this this supposed to include vectors for main components of MWE only if this synset has no single_word representation or if MWE is found in vectors
        elif mode == 'main':
            if len(name.split()) == 1:
                if tags == True:
                    if pos == 'NOUN':
                        item = name.lower() + '_NOUN'
                    elif pos == 'VERB':
                        item = name.lower() + '_VERB'
                    else:
                        item = None
                        print('Which PoS part of WordNet are we dealing with?')
                else:
                    item = name.lower()
                
                all_id_senses.append((id, item))
            
            ## TODO: apply this condition only to synsets with no single_word representation;
            if len(name.split()) > 1:
                item = preprocess_mwe(name, tags=tags,
                                      pos=pos)  ## this is compatible with embeddings already (lower, tagged)
                if item in emb_voc:
                    ### adding the few cases of MWE in embeddings vocabulary
                    all_id_senses.append((id, item))
                else:
                    ## only if the respective synset has not been already added; no garantee that it has no single word lemmas further down
                    count_main += 1
                    if tags == True:
                        if pos == 'NOUN':
                            item = main_wd.lower() + '_NOUN'
                        elif pos == 'VERB':
                            item = main_wd.lower() + '_VERB'
                        else:
                            item = None
                            print('Which PoS part of WordNet are we dealing with?')
                    else:
                        item = main_wd.lower()
                    
                    all_id_senses.append((id, item))
                    ## TODO deduplicate tuples
        else:
            print('What do you want to do with senses that are lexicalised as MWE?')
    return all_id_senses ## 134530-N, кунгур_NOUN

def lemmas_based_hypers(test_item, vec=None, emb=None, topn=None, dict_w2ids=None): #кунгур_NOUN:['134530-N']
    nosamename = 0
    hyper_vec = np.array(vec, dtype=float)
    temp = set()
    deduplicated_sims = []
    nearest_neighbors = emb.most_similar(positive=[hyper_vec], topn=topn)
    sims = []
    for res in nearest_neighbors:
        hypernym = res[0]
        similarity = res[1]
        if hypernym in dict_w2ids:
            for synset in dict_w2ids[hypernym]:  # we are adding all synset ids associated with the topN most_similar in embeddings and found in ruWordnet
                sims.append((synset, hypernym, similarity))
    
    # sort the list of tuples (id, sim) by the 2nd element and deduplicate
    # by rewriting the list while checking for duplicate synset ids
    sims = sorted(sims, key=itemgetter(2), reverse=True)
    
    for a, b, c in sims:
        # exclude same word as hypernym
        b = b[:-5].upper()
        if test_item != b:
            # print(hypo, b)
            if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b, c))
        else:
            nosamename += 1
    # print('Selves as hypernyms: %s' % nosamename)
    this_hypo_res = deduplicated_sims[:10]  # list of (synset_id, hypernym_word, sim)
    
    return this_hypo_res

def synsets_vectorized(emb=None, worded_synsets=None, named_synsets=None, tags=None, pos=None):
    total_lemmas = 0
    single_lemmas = 0
    ruthes_oov = 0
    mean_synset_vecs = []
    synset_ids_names = []
    for id, wordlist in worded_synsets.items():
        # print('==', id, named_synsets[id], wordlist)
        current_vector_list = []
        for w in wordlist:
            total_lemmas += 1
            if len(w.split()) == 1:
                single_lemmas += 1
                if tags:
                    if pos == 'NOUN':
                        w = (w.lower() + '_NOUN')
                    if pos == 'VERB':
                        w = (w.lower() + '_VERB')
                if w in emb.vocab:
                    # print('++', w, emb[w])
                    current_vector_list.append(emb[w])
                    current_array = np.array(current_vector_list)
                    this_mean_vec = np.mean(current_array,axis=0)  # average column-wise, getting a new row=vector of size 300
                    mean_synset_vecs.append(this_mean_vec)
                    synset_ids_names.append((id, named_synsets[id]))
                else:
                    ruthes_oov += 1
                    this_mean_vec = None
            else:
                continue

    print('===300===', len(this_mean_vec))
    print('Total lemmas: ', total_lemmas)
    print('Singleword lemmas: ', single_lemmas)
    print('Singleword lemmas in ruWordNet absent from embeddings: ', ruthes_oov)
    ## synset_ids_names has 134530-N, КУНГУР
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
        # b = b[:-5].upper()
        if test_item != b:
            # print(hypo, b)
            if a not in temp:
                temp.add(a)
                deduplicated_sims.append((a, b, c))
        else:
            nosamename += 1

    this_hypo_res = deduplicated_sims[:topn]  ## list of (synset_id, hypernym_word, sim)
    
    return this_hypo_res  # list of (synset_id, hypernym_synset_name, sim)

## кунгур_NOUN:['134530-N'], агностик_NOUN	["атеист_NOUN", "человек_NOUN", "религия_NOUN", ...]
def cooccurence_counts(test_item, vec=None, emb=None, topn=None, dict_w2ids=None, corpus_freqs=None):
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
    
    new_list = []
    if len(corpus_freqs[test_item]) != 0:

        for i in corpus_freqs[test_item]: # [word_NOUN, word_NOUN, word_NOUN]
            for tup in deduplicated_sims[:25]:  ## maybe further limit these 500 words to 100?
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


def get_random_test(goldpath=None, w2ids_d=None, method=None):
    gold_dict = defaultdict(list)

    gold = open(goldpath, 'r').readlines()
    print(goldpath)
    for id, line in enumerate(gold):
        # skip the header
        if id == 0:
            continue
        
        pair = line.split('\t')
        # print(pair)
        if method == 'deworded':
            hypo = pair[0].strip()
            hyper = pair[1].strip()
            gold_dict[hypo].append(hyper)  ## the values is list of lists of ids in the deworded method
        else:
            hypo = pair[0].strip()
            hypo = hypo[:5].upper()
            hyper = pair[1].strip()
            hyper = hyper[:5].upper()
            ## replace the worded golden hypernym with all synset_ids possible for this hypernym
            syn_ids = w2ids_d[hyper]  ## a list of hypernym synset_ids
            gold_dict[hypo].append(syn_ids)
    
    if method != 'deworded':
        gold_dict0 = defaultdict(list)
        for key in gold_dict:
            gold_dict0[key] = [item for sublist in gold_dict[key] for item in sublist]  # flatten the list
    
        first2pairs_gold = {k: gold_dict0[k] for k in list(gold_dict0)[:10]}
        
        print(first2pairs_gold)
        print(len(gold_dict0))
    
        return gold_dict0
    
    else:
        first2pairs_gold = {k: gold_dict[k] for k in list(gold_dict)[:10]}
    
        print(first2pairs_gold)
        print(len(gold_dict))
        
        return gold_dict


def get_intrinsic_test(goldpath=None):
    gold_dict = defaultdict(list)
    print('Evaluating on the intrinsic testset')
    df_test = pd.read_csv(goldpath, sep='\t')
    df_test = df_test.replace(to_replace=r"[\[\]']", value='', regex=True)
    
    ids = df_test['SYNSET_ID'].tolist()
    my_TEXTS = df_test['TEXT'].tolist()
    ids_parents = df_test['PARENTS'].tolist()
    
    for hypo, hyper in zip(my_TEXTS, ids_parents):
        hypo = hypo.replace(r'"', '')
        hyper = hyper.replace(r'"', '')
        hypo = hypo.split(', ')
        hyper = hyper.split(', ')
        print(hyper)
        for i in hypo:
            gold_dict[i].append(hyper)
    
    gold_dict0 = defaultdict(list)
    for key in gold_dict:
        gold_dict0[key] = [item for sublist in gold_dict[key] for item in sublist]  # flatten the list
    
    first2pairs_gold = {k: gold_dict0[k] for k in list(gold_dict0)[:2]}
    
    print(first2pairs_gold)
    
    return gold_dict0


######################


if __name__ == '__main__':
    print('=== This is a modules script, it is not supposed to run as main ===')
