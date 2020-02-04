import csv
import os
from xml.dom import minidom
from collections import defaultdict
from collections import Counter

from hyper_imports import read_xml

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
    print(len(set(all_ids)))
    
    ratios = defaultdict(int)
    for id in all_ids:
        try:
            ratios[id] = freq_hyper[id]/freq_hypo[id]
        except ZeroDivisionError:
            continue

    sort_it = {k: v for k, v in sorted(ratios.items(), key=lambda item: item[1], reverse=True)}
    # for id in sort_it:
    #     print(id, sort_it[id])
    my_ten = []
    for i, (k,v) in enumerate(sort_it.items()):
        if i < 10:
            my_ten.append(k)
            print(k)

    return my_ten ## synsets ids
    
        
        

if __name__ == '__main__':
    
    rel_path = '/home/u2/resources/hypohyper/ruwordnet/synset_relations.N.xml'
    top_ten = popular_generic_concepts(rel_path)