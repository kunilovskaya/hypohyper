#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
import csv
import os
import pickle
import sys
from gzip import GzipFile
from itertools import zip_longest
from smart_open import open
import numpy as np
import tensorflow as tf
from helpers import load_embeddings

from projlearn import *

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


parser = argparse.ArgumentParser(description='Prediction.')
parser.add_argument('--w2v', required=True)
parser.add_argument('--kmeans', default='kmeans.pickle', nargs='?', help='Path to k-means.pickle.')
parser.add_argument('--model', default='baseline', nargs='?', choices=MODELS.keys(), help='The model.')
parser.add_argument('--path', default='', nargs='?', help='The path to the model dump.')
parser.add_argument('--slices', default=100000, type=int, help='The slice size.')
parser.add_argument('output', help='Output file.')
args = parser.parse_args()

w2v = load_embeddings(args.w2v)

print('Using %d word2vec dimensions from "%s".' % (w2v.vector_size, args.w2v), flush=True, file=sys.stderr)

kmeans = pickle.load(open(args.kmeans, 'rb'))

print('The number of clusters is %d.' % kmeans.n_clusters, flush=True, file=sys.stderr)

model = MODELS[args.model](x_size=w2v.vector_size, y_size=w2v.vector_size, w_stddev=0, lambda_=0)

reader = csv.reader(sys.stdin, delimiter='\t', quoting=csv.QUOTE_NONE)

predictions = [[], [], []]

with open(args.output, 'wb') as f:
    for s, rows in enumerate(grouper(args.slices, reader)):
        X_all, Y_all = [], []

        for row in rows:
            if row is None:
                continue

            predictions[0].append(row[0])
            predictions[1].append(row[1])

            if row[0] in w2v.vocab:
                X_all.append(w2v.wv.syn0[w2v.wv.vocab[row[0]].index])
            else:
                print(row[0], 'out of vocabulary!', file=sys.stderr)
                X_all.append(np.zeros(w2v.vector_size))
            if row[1] != 'None':
                Y_all.append(w2v.wv.syn0[w2v.wv.vocab[row[1]].index])
            else:
                Y_all.append(np.zeros(w2v.vector_size))

        # X_all, Y_all = w2v.wv.syn0[X_all], w2v.wv.syn0[Y_all]
        X_all, Y_all = np.array(X_all), np.array(Y_all)

        offsets = Y_all - X_all

        X_clusters_list = list(enumerate(kmeans.predict(offsets)))

        X_clusters = {}

        for cluster in range(kmeans.n_clusters):
            X_clusters[cluster] = [i for i, c in X_clusters_list if c == cluster]

        Y_hat_all = np.empty(X_all.shape)

        for cluster, indices in X_clusters.items():
            with tf.compat.v1.Session() as sess:
                saver = tf.compat.v1.train.Saver()

                saver.restore(sess, os.path.join(args.path, '%s.k%d.trained') % (args.model, cluster + 1))

                Y_hat = sess.run(model.Y_hat, feed_dict={model.X: X_all[indices]})

                for i, j in enumerate(indices):
                    if not np.any(X_all[j]):
                        Y_hat_all[j] = X_all[j]
                    else:
                        Y_hat_all[j] = Y_hat[i]
                    nearest_neighbors = w2v.most_similar(positive=[Y_hat[i]])
                    predictions[2].append([i[0] for i in nearest_neighbors])

        for inp, gold, predicted in zip (predictions[0], predictions[1], predictions[2]):
            print('\t'.join([inp, gold]), predicted)
        np.save(f, Y_hat_all, allow_pickle=False)

        print('%d slices done.' % (s + 1), flush=True, file=sys.stderr)
