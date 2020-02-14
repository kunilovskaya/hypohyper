#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
import csv
import operator
import pickle
import random
import sys
from multiprocessing import Pool
from smart_open import open
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser(description='Clustering.')
parser.add_argument('--train', default='train.npz', nargs='?', help='Path to the training set.')
parser.add_argument('--seed', default=228, type=int, nargs='?', help='Random seed.')
parser.add_argument('-k', type=int, nargs='?', help='Number of clusters.')
args = vars(parser.parse_args())

RANDOM_SEED = args['seed']
random.seed(RANDOM_SEED)

with np.load(args['train']) as npz:
    X_index_train = npz['X_index']
    Y_all_train = npz['Y_all']
    Z_all_train = npz['Z_all']

X_all_train = Z_all_train[X_index_train[:, 0], :]

train_offsets = Y_all_train - X_all_train

if args['k']:
    km = KMeans(n_clusters=args['k'], n_jobs=4, random_state=RANDOM_SEED)
    km.fit_predict(train_offsets)
    pickle.dump(km, open('kmeans.pickle', 'wb'))
    print('Just written the k-means result for k=%d.' % (km.n_clusters))
    sys.exit(0)

kmeans = {}

for k in range(2, 20 + 1):
    kmeans[k] = KMeans(n_clusters=k, n_jobs=4, random_state=RANDOM_SEED)
    kmeans[k].fit_predict(train_offsets)
    print('k-means for k=%d computed.' % (k))


def evaluate(k):
    km = kmeans[k]
    score = silhouette_score(train_offsets, km.labels_, metric='euclidean', random_state=RANDOM_SEED)
    print('Silhouette score for k=%d is %f.' % (k, score))
    return (k, score)


scores = {}

with open('kmeans-scores.txt', 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
    writer.writerow(('k', 'silhouette'))
    with Pool(12) as pool:
        for k, score in pool.imap_unordered(evaluate, kmeans):
            scores[k] = score
            writer.writerow((k, score))

k, score = max(scores.items(), key=operator.itemgetter(1))
pickle.dump(kmeans[k], open('kmeans.pickle', 'wb'))
