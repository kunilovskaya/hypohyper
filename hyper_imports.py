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
                    # if hypo in embedding.vocab and hyper in embedding.vocab:
                    #     smaller_train.append((hypo, hyper))
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
                ## this is tuned for ft vectors to filter out OOV (mostly MWE)
                if skip_oov == False:
                    smaller_train.append((hypo.lower(), hyper.lower()))
                elif skip_oov == True:
                    if hypo.lower() in embedding.vocab and hyper.lower() in embedding.vocab:
                        smaller_train.append((hypo.lower(), hyper.lower()))

    return smaller_train  # only the pairs that are found in the embeddings if skip_oov=True


def write_hyp_pairs(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='unix', delimiter='\t', lineterminator='\n')
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
    # print(all_ids[:5])
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
            print(k)
    
    
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
    return all_id_senses

if __name__ == '__main__':
    print('=== This is a modules script, it is not supposed to run as main ===')
