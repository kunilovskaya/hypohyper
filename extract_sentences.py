#! python3
# coding: utf-8

from argparse import ArgumentParser
import sys
from smart_open import open

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--words', required=True, help="path to word list")
    args = parser.parse_args()

    words = set()

    for line in open(args.words):
        word = line.strip().lower()
        word = word.replace(' ', '::')
        words.add(word)

    print('%d words read' % len(words), file=sys.stderr)

    for line in sys.stdin:
        res = set(line.strip().split())
        for w in words:
            if w in res:
                print(line)
                break


