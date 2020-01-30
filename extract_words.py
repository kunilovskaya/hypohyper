#! python3
# coding: utf-8

from hyper_import_functions import read_xml, id2wds_dict, preprocess_mwe
from argparse import ArgumentParser
import sys
from smart_open import open

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--synsets', required=True, help="path to RuWordNet XML")
    parser.add_argument('--out', help="path to save extracted words")
    args = parser.parse_args()

    parsed_syns = read_xml(args.synsets)

    synsets = id2wds_dict(parsed_syns)

    synsets = {el: [preprocess_mwe(word) for word in synsets[el]] for el in synsets}

    print('Extracted %d synsets' % len(synsets), file=sys.stderr)

    words = set([word for el in synsets for word in synsets[el]])

    print('Extracted %d words' % len(words), file=sys.stderr)

    if args.out:
        with open(args.out, 'a') as f:
            for word in sorted(words):
                f.write(word + '\n')
