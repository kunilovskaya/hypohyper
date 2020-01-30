import csv
import os
from xml.dom import minidom
from collections import defaultdict
from collections import Counter
import pandas as pd
import logging
from gensim.models import KeyedVectors
import numpy as np
from gensim.matutils import unitvec
from smart_open import open


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
        identifier = syn.getAttributeNode('id').nodeValue
        name = syn.getAttributeNode('ruthes_name').nodeValue

        id2name[identifier] = name

    return id2name


def wd2id_dict(id2dict):
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


def read_train(tsv_in):
    df_out = pd.read_csv(tsv_in, sep='\t')
    return df_out


def load_embeddings(modelfile):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
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
    else:  # Native Gensim format?
        emb_model = KeyedVectors.load(modelfile)
    emb_model.init_sims(replace=True)
    print('Success! Vectors loaded')

    return emb_model


def get_vector(word, emb=None):
    if not emb:
        return None
    vector = emb[word]
    return vector


def preprocess_mwe(item, tags=False):
    # Alas, those bigrams are overwhelmingly proper names while we need multi-word concepts.
    # For example, in aranea: "::[а-я]+\_NOUN" 8369 item, while the freq of all "::" 8407
    if len(item.split()) > 1:
        item = '::'.join(item.lower().split())
        if tags:
            item = item + '_PROPN'
        # print(item)
    else:
        item = item.lower()
        if tags:
            item = item + '_NOUN'

    return item

## now this function stores lowercased word pairs regardless of the combination of tags/mwe boolean options)
def filter_dataset(pairs, embedding, tags=None, mwe=None):
    smaller_train = []
    for hypo, hyper in pairs:
        if tags:
            if mwe: ## this returns lowercased words
                hypo = preprocess_mwe(hypo, tags=True)
                hyper = preprocess_mwe(hyper, tags=True)
                if hypo in embedding and hyper in embedding:
                    smaller_train.append((hypo, hyper))
            else:
                if hypo.lower() + '_NOUN' in embedding and hyper.lower() + '_NOUN' in embedding:
                    ## what does the line below do?
                    if hypo in embedding and hyper in embedding:
                        smaller_train.append((hypo.lower(), hyper.lower()))
        else:
            if mwe: ## this returns lowercased words
                hypo = preprocess_mwe(hypo)
                hyper = preprocess_mwe(hyper)
            ## there should be else below?
            if hypo.lower() in embedding and hyper.lower() in embedding:
                smaller_train.append((hypo.lower(), hyper.lower()))

    return smaller_train  # only the pairs that are found in the embeddings


def write_hyp_pairs(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='unix', delimiter='\t', lineterminator='\n')
        writer.writerow(['hyponym', 'hypernym'])
        for pair in data:
            writer.writerow(pair)


def learn_projection(dataset, embedding, lmbd=1.0, from_df=False):
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


if __name__ == '__main__':
    print('=== This is a modules script, it is not supposed to run as main ===')
