import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser

from smart_open import open
from hyper_imports import preprocess_mwe
from configs import VECTORS, RUWORDNET, OUT, POS, TEST, METHOD


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--words', default='output/ruWordNet_lemmas.txt',
                        help="tagged 68K words and phrases from ruWordNet")
    # parser.add_argument('--words', default='output/mwe_support/ruwordnet_names_pos.txt', help="tagged 68K words and phrases from ruWordNet")
    args = parser.parse_args()

    words = set()

    for line in open(args.words):
        word = line.strip()
        word = preprocess_mwe(word, tags=True, pos=POS)
        words.add(word)

    print('%d words read' % len(words), file=sys.stderr)

    for line in sys.stdin: # zcat corpus.txt.gz | python3 find_words.py
        res = set(line.strip().split())
        for w in words:
            if w in res:
                print(line) ## this is fed into another script
                break



## in line from stdin lose tags