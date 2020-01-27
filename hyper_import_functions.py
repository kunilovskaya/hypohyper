import csv

from xml.dom import minidom
from collections import defaultdict
from collections import Counter
import pandas as pd

from gensim.models import KeyedVectors


# parse ruwordnet
# and (1) build one-to-one dicts for synsets_ids and any asociated lemmas/=words/=senses/=synonyms
def read_xml(xml_file):
    doc = minidom.parse(xml_file)
    # node = doc.documentElement
    try:
        parsed = doc.getElementsByTagName("synset") or doc.getElementsByTagName("relation")  # it is a list
    except:
        # nodelist = parsed
        print('Are you sure you are passing the expected files?')
        parsed = None
    
    return parsed ## a list of xml entities



## how does this account for the fact that wd can be a member of several synsets?: synset2wordlist mappings are unique, while the reverse is not true
def id2wds_dict(synsets):
    id2wds = defaultdict(list)
    for syn in synsets:
        id = syn.getAttributeNode('id').nodeValue
        senses = syn.getElementsByTagName("sense")
        for sense in senses:
            wd = sense.childNodes[-1].data
            id2wds[id].append(wd)
    
    return id2wds  ## get a dict of format 144031-N:[АУТИЗМ, АУТИСТИЧЕСКОЕ МЫШЛЕНИЕ]


## map synset ids to synset names from synsets.N.xml (ex. ruthes_name="УЛЫБКА")
def id2name_dict(synsets):
    id2name = defaultdict()
    for syn in synsets:
        id = syn.getAttributeNode('id').nodeValue
        name = syn.getAttributeNode('ruthes_name').nodeValue

        id2name[id] = name
    
    return id2name

def wd2id_dict(id2dict):
    wd2id = defaultdict(list)
    for k, values in id2dict.items():
        for v in values:
            wd2id[v].append(k)
    
    return wd2id  ## get a dict of format, where values are lists of synset_ids for each word; ex. АУТИЗМ:[144031-N], АУТИСТИЧЕСКИЙ МЫШЛЕНИЕ:[144031-N]


## distribution of relations annotated in ruwordnet
def get_all_rels(relations):
    my_list = []
    for rel in relations:
        rel_name = rel.getAttributeNode('name').nodeValue
        my_list.append(rel_name)
    my_dict = Counter(my_list) #Counter({'apple': 3, 'egg': 2, 'banana': 1})
   
    # iterator = iter(my_dict.items())
    # for i in range(len(my_dict)):
    #     print(next(iterator))
    print('=====\n %s \n=====' % my_dict)

#(2) по лемме (ключи в нашем словаре) его синсет-гипероним и синсет-гипоним как списки лемм и номера соответствующих синсетов
## if you pass hyponym you get the query hyperonyms (ids)
def get_rel_by_name(relations, word, wds2ids, id2wds, name=None): ## level: hypernym, hyponym, domain, POS-synonymy, instance hypernym, instance hyponym
    these_ids = wds2ids[word] ## this is a list of possible ids for this word
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

def get_rel_by_synset_id(relations, id, id2wds, name=None):  ## level: hypernym, hyponym, domain, POS-synonymy, instance hypernym, instance hyponym
    wds_in_this_id = id2wds[id]
    related_ids = []
    related_wds = []
    print('Inspecting relations for the synset: %s' % wds_in_this_id)
    for rel in relations:
        parent = rel.getAttributeNode('parent_id').nodeValue
        child = rel.getAttributeNode('child_id').nodeValue
        rel_name = rel.getAttributeNode('name').nodeValue
        if id == child and name == rel_name:
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
    if modelfile.endswith('.txt.gz') or modelfile.endswith('.txt') or modelfile.endswith('.vec.gz'):
        model = KeyedVectors.load_word2vec_format(modelfile, binary=False, encoding='utf-8', unicode_errors='ignore')
    elif modelfile.endswith('.bin.gz') or modelfile.endswith('.bin') or modelfile.endswith('.w2v'):
        model = KeyedVectors.load_word2vec_format(modelfile, binary=True, encoding='utf-8', unicode_errors='ignore', limit=4000000)
    else:
        model = KeyedVectors.load(modelfile)  # For newer models
    model.init_sims(replace=True)
    print('Success! Vectors loaded')
    return model


def get_vector(word, emb=None):
    if not emb:
        return None
    vector = emb[word]
    return vector


def preprocess_mwe(item):
    ## Alas, those bigrams are overwhelmingly proper names while we need multi-word concepts.
    # For example, in aranea: "::[а-я]+\_NOUN" 8369 item, while the freq of all "::" 8407
    if len(item.split()) >=2:
        item = '::'.join(item.split())
        item = item.lower()+'_PROPN'
        # print(item)
    else:
        item = item.lower()+'_NOUN'
        
    return item

def filter_dataset(pairs, embedding, tags=None, mwe=None):
    smaller_train = []
    for hypo,hyper in pairs:

        if tags == True:
            if mwe == True:
                hypo = preprocess_mwe(hypo)
                hyper = preprocess_mwe(hyper)
                if hypo in embedding and hyper in embedding:
                    smaller_train.append((hypo, hyper))
            elif mwe == False:
                if hypo.lower()+'_NOUN' in embedding and hyper.lower()+'_NOUN' in embedding:
                    if hypo in embedding and hyper in embedding:
                        smaller_train.append((hypo, hyper))
        elif tags == False:
            if hypo.lower() in embedding and hyper.lower() in embedding:
                smaller_train.append((hypo, hyper))
    
    return smaller_train ## only the pairs that are found in the embeddings
    
    
def write_hyp_pairs(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
        for pair in data:
            writer.writerow(pair)
            


## python3 mappings.py --relations /home/u2/resources/ruwordnet/synset_relations.N.xml --synsets /home/u2/resources/ruwordnet/synsets.N.xml --train /home/u2/data/hypohyper/training_data/training_nouns.tsv
if __name__ == '__main__':
    print('=== This is a modules script, it is not supposed to run as main ===')
        

    

    