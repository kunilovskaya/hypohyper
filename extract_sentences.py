#! python3
# coding: utf-8

from argparse import ArgumentParser
import sys
from smart_open import open
from hyper_imports import preprocess_mwe


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--words', required=True, help="path to word list")
    parser.add_argument('--pos', default='NOUN', help="PoS tag to use")
    args = parser.parse_args()

    words = set()

    for line in open(args.words):
        word = line.strip()
        word = preprocess_mwe(word, tags=True, pos=args.pos) 
        words.add(word)

    print('%d words read' % len(words), file=sys.stderr)

    for line in sys.stdin:
        res = set(line.strip().split())
        for w in words:
            if w in res:
                print(line)
                break


