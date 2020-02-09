#! python3
# coding: utf-8

from argparse import ArgumentParser
import sys
from smart_open import open
from hyper_imports import preprocess_mwe
import json

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--words', required=True, help="path to input word list")
    parser.add_argument('--synsets', required=True, help="path to words from WordNet")
    parser.add_argument('--pos', default='NOUN', help="PoS tag to use")
    args = parser.parse_args()

    words = {}

    for line in open(args.words):
        word = preprocess_mwe(line.strip(), tags=True, pos=args.pos)
        words[word] = {}

    synset_words = set()

    for line in open(args.synsets):
        word = preprocess_mwe(line.strip(), tags=True, pos=args.pos)
        synset_words.add(word)
    print('%d words read' % len(words), file=sys.stderr)
    print('%d synset words read' % len(synset_words), file=sys.stderr)

    for line in sys.stdin:
        lemmas = set(line.strip().split())
        for word in words:
            if word in lemmas:
                for synset_word in synset_words:
                    if synset_word in lemmas:
                        if synset_word not in words[word]:
                            words[word][synset_word] = 0
                        words[word][synset_word] += 1

    print('We found data for %d input words' % len([w for w in words if len(words[w]) > 1]), file=sys.stderr)
    out = json.dumps(words, ensure_ascii=False, indent=4, sort_keys=True)
    print(out)
