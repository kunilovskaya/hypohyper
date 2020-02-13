#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
import csv
import random
from collections import defaultdict
import numpy as np
from gensim.models.word2vec import Word2Vec
from helpers import load_embeddings
from smart_open import open

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

parser = argparse.ArgumentParser(description='Preparation.')
parser.add_argument('--w2v', default='all.norm-sz100-w10-cb0-it1-min100.w2v', nargs='?',
                    help='Path to the word2vec model.')
parser.add_argument('--seed', default=228, type=int, nargs='?', help='Random seed.')
args = vars(parser.parse_args())

RANDOM_SEED = args['seed']
random.seed(RANDOM_SEED)

w2v = load_embeddings(args['w2v'])
print('Using %d word2vec dimensions from "%s".' % (w2v.vector_size, args['w2v']))


def read_subsumptions(filename):
    subsumptions = []

    with open(filename) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            subsumptions.append((row[0], row[1]))

    return subsumptions


def read_synonyms(filename):
    synonyms = defaultdict(lambda: list())

    with open(filename) as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            for word in row[1].split(','):
                synonyms[row[0]].append(word)

    return synonyms


subsumptions_train = read_subsumptions('subsumptions-train.txt.gz')
subsumptions_validation = read_subsumptions('subsumptions-validation.txt.gz')
subsumptions_test = read_subsumptions('subsumptions-test.txt.gz')
synonyms = read_synonyms('synonyms.txt.gz')


def compute_XZ(subsumptions):
    X_index, Z_all = [], []

    for hyponym, hypernym in subsumptions:
        offset = len(Z_all)
        word_synonyms = [hyponym] + synonyms[hyponym]

        X_index.append([offset, len(word_synonyms)])

        for synonym in word_synonyms:
            Z_all.append(w2v[synonym])

    return (np.array(X_index, dtype='int32'), np.array(Z_all))


X_index_train, Z_all_train = compute_XZ(subsumptions_train)
X_index_validation, Z_all_validation = compute_XZ(subsumptions_validation)
X_index_test, Z_all_test = compute_XZ(subsumptions_test)

Y_all_train = np.array([w2v[w] for _, w in subsumptions_train])
Y_all_validation = np.array([w2v[w] for _, w in subsumptions_validation])
Y_all_test = np.array([w2v[w] for _, w in subsumptions_test])

np.savez_compressed('train.npz', X_index=X_index_train,
                    Y_all=Y_all_train,
                    Z_all=Z_all_train)

np.savez_compressed('validation.npz', X_index=X_index_validation,
                    Y_all=Y_all_validation,
                    Z_all=Z_all_validation)

np.savez_compressed('test.npz', X_index=X_index_test,
                    Y_all=Y_all_test,
                    Z_all=Z_all_test)

print('I have %d train, %d validation and %d test examples.' % (
    Y_all_train.shape[0],
    Y_all_validation.shape[0],
    Y_all_test.shape[0]))
