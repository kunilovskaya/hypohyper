#! python3
# coding: utf-8
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from hyper_imports import read_xml, id2wds_dict, preprocess_mwe
from argparse import ArgumentParser

from smart_open import open
from configs import OUT, RUWORDNET

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--synsets', default='%ssynsets.N.xml' % RUWORDNET, help="synsets files") ## has lemmatised senses
    parser.add_argument('--out', default='%sruWordNet_words.txt' % OUT, help="path to save extracted lemmas")
    args = parser.parse_args()

    parsed_syns = read_xml(args.synsets)
    synsets = id2wds_dict(parsed_syns)

    synsets = {el: [preprocess_mwe(word) for word in synsets[el]] for el in synsets}

    print('Extracted %d synsets' % len(synsets), file=sys.stderr)

    words = set([word for el in synsets for word in synsets[el]])

    print('Extracted %d words' % len(words), file=sys.stderr)

    with open(args.out_lemmas, 'a') as f:
        for word in sorted(words):
            f.write(word + '\n')
