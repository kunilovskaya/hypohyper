#! python3
# coding: utf-8
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser
from smart_open import open
import json
from trials_errors.hyper_imports import read_xml, id2wds_dict, preprocess_mwe
from trials_errors.configs import VECTORS, OUT, POS, TEST

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--datafile', default='%scooc_%s_%s_%s.json' % (OUT, VECTORS, POS, TEST), help="JSON with co-occurrence data")
    parser.add_argument('--wordnet', default=None,
                        help="path to WordNet XML file") #default='input/resources/ruwordnet/synsets.N.xml',
    parser.add_argument('--pos', default='NOUN', help="PoS tag to use")
    args = parser.parse_args()

    with open(args.datafile, 'rb') as f:
        data = json.loads(f.read())

    w2syn = {}
    if args.wordnet:
        parsed_syns = read_xml(args.wordnet)
        synsets = id2wds_dict(parsed_syns)
        for synset in synsets:
            for word in synsets[synset]:
                conv_word = preprocess_mwe(word, tags=True, pos=args.pos)
                if conv_word not in w2syn:
                    w2syn[conv_word] = set()
                w2syn[conv_word].add(synset)

    for word in data:
        predictions = [w for w in sorted(data[word], key=data[word].get, reverse=True)]
        if args.wordnet:
            counter = 0
            hypernyms = []
            seen_synsets = set()
            for p in predictions:
                if p == 'год_NOUN' or p == 'время_NOUN':  # too general concepts
                    continue
                if counter > 9:
                    break
                pred_synsets = w2syn[p]
                for synset in pred_synsets:
                    if counter > 9:
                        break
                    if synset not in seen_synsets:
                        out = [word.upper().split('_')[0], synset, p.split('_')[0]]
                        # print('\t'.join(out))
                        counter += 1
                    seen_synsets.add(synset)
        else:
            hypernyms = json.dumps(predictions, ensure_ascii=False)
            print('\t'.join([word, hypernyms]))
