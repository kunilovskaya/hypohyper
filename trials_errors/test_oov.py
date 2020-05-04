#! python3
# coding: utf-8

from trials_errors.hyper_imports import load_embeddings
from argparse import ArgumentParser
import sys
from smart_open import open

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--words', required=True, help="path to word list")
    parser.add_argument('--model', help="path to embeddings")
    args = parser.parse_args()

    words = set()

    for line in open(args.words):
        word = line.strip().lower()
        word = word.replace(' ', '::')
        words.add(word)

    print('%d words read' % len(words), file=sys.stderr)

    model = load_embeddings(args.model)

    vocab = set(model.wv.index2word)

    print('Model contains %d words in its vocabulary' % len(vocab), file=sys.stderr)

    oov = words - vocab

    for word in oov:
        if '::' not in word:
            print(word)

    print('Found %d OOV words' % len(oov), file=sys.stderr)
