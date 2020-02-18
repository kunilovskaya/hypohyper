import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)
# alternatively
# sys.path.append('/home/u2/git/hypohyper/')

from hyper_imports import read_xml, id2wds_dict, preprocess_mwe
from argparse import ArgumentParser

from xml.dom import minidom
from smart_open import open
from configs import OUT, RUWORDNET

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--senses', default='%ssenses.N.xml' % RUWORDNET, help="files with non-lemmatised names of senses")
    parser.add_argument('--out_lemmas', default='%sruWordNet_lemmas.txt' % OUT, help="path to save extracted lemmas")
    parser.add_argument('--out_names', default='%sruWordNet_names.txt' % OUT, help="path to save extracted words")
    args = parser.parse_args()

    parsed_senses = read_xml(args.senses)
            
    names = set() ## testing this way of deduplication
    lemmas = []
    ids = []
    doc = minidom.parse(args.senses)
    parsed_senses = doc.getElementsByTagName("sense")
    for sense in parsed_senses:
        # print(sense)
        id = sense.getAttributeNode('synset_id').nodeValue
        name = sense.getAttributeNode("name").nodeValue  # name="НАРУШЕНИЕ ПРАВИЛ ПРОДАЖИ"
        lemma = sense.getAttributeNode("lemma").nodeValue
        names.add(name)
        lemmas.append(lemma)
        ids.append(id)
        
    
    
    print('Extracted %d ids' % len(set(ids)), file=sys.stderr)
    
    with open(args.out_names, 'a') as f_names:
        for i in names:
            f_names.write(i.strip() + '\n')
    print('Extracted %d names' % len(names), file=sys.stderr)
    
    with open(args.out_lemmas, 'a') as f_lemmas:
        for i in set(lemmas):
            f_lemmas.write(i.strip() + '\n')
    print('Extracted %d lemmas' % len(set(lemmas)), file=sys.stderr)

