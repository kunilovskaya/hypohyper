#! python
# coding: utf-8

from collections import Counter
import os
import random
import json
import time
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from hyper_imports import load_embeddings
from configs import OUT, POS, EMB_PATH, VECTORS, TEST

if __name__ == '__main__':
    parser = ArgumentParser()
    if POS == 'NOUN' and 'mwe' in VECTORS:
        if TEST == 'provided':
            parser.add_argument('--path', default='input/data/notest_newtags_all_data_nouns.tsv',
                                help="Path to the training corpus compatible with the vectors")
        else:
            parser.add_argument('--path', default='input/data/newtags_all_data_nouns.tsv',
                                help="Path to the training corpus compatible with vectors format")
    if POS == 'NOUN' and 'mwe' not in VECTORS:
        if TEST == 'provided':
            parser.add_argument('--path', default='input/data/notest_oldtags_all_data_nouns.tsv',
                                help="Path to the training corpus compatible with the vectors")
        else:
            parser.add_argument('--path', default='input/data/oldtags_all_data_nouns.tsv',
                                help="Path to the training corpus compatible with the vectors")
    if POS == 'VERB':
        parser.add_argument('--path', default='input/data/newtags_all_data_VERB.tsv',
                            help="Path to the training corpus")

    parser.add_argument('--w2v', default=EMB_PATH, help="Path to the embeddings")
    parser.add_argument('--hidden_dim', action='store', type=int, default=386)
    parser.add_argument('--batch_size', action='store', type=int, default=32)
    parser.add_argument('--run_name', default='notest_' + VECTORS + '_' + POS,
                        help="Human-readable name of the run.")
    parser.add_argument('--epochs', action='store', type=int, default=25)
    parser.add_argument('--freq', action='store', type=int, default=5,
                        help="Frequency threshold for synsets")
    parser.add_argument('--split', action='store', type=float, default=0.9)
    args = parser.parse_args()

    trainfile = args.path
    run_name = args.run_name

    # We don't need a separate dev/dalidation dataset since it is created automatically
    # by TF from the train set.

    # Fix random seeds for repeatability of experiments:
    random.seed(42)
    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)

    print('Loading the dataset...')
    train_dataset = pd.read_csv(trainfile, sep='\t', header=0)
    print('Finished loading the dataset')

    BALANCED = False  # Do we want to address the issue of imbalanced class weights?
    OVERFIT = True  # Do we want to discard early stopping and train until convergence?

    embedding = load_embeddings(args.w2v)

    x_train = []
    y_train = []

    oov = 0
    mwe_oov = 0

    train_synsets = Counter()
    for target in train_dataset['synsets']:
        synsets = json.loads(target)
        train_synsets.update(synsets)  # flattens the list; gets a list of all synset ids

    valid_synsets = set([s for s in train_synsets if train_synsets[s] > args.freq])

    for word, target in zip(train_dataset['word'], train_dataset['synsets']):
        lemma = word
        if lemma in embedding.vocab:
            vector = embedding[lemma]
            synsets = json.loads(target)
            for synset in synsets:
                if synset in valid_synsets:
                    # word vector with each hypernym synset as the class label:
                    x_train.append(vector)
                    y_train.append(synset)
        else:
            oov += 1
            if '::' in lemma:
                mwe_oov += 1

    print(len(x_train), 'train instances')
    print('%d OOV instances, including %d MWE' % (oov, mwe_oov))

    classes = sorted(list(set(y_train)))
    num_classes = len(classes)
    print(num_classes, 'classes')

    class_weights = {}
    if BALANCED:
        for nr, synset in enumerate(classes):
            class_weights[nr] = 1 / train_synsets[synset]

    targets = Counter(y_train)

    print('===========================')
    print('Class distribution in the training data:')
    print(targets.most_common(10))
    print('===========================')

    x_train = np.array(x_train)

    print('Train data shape:', x_train.shape)

    # Converting text labels to indexes
    y_train = [classes.index(i) for i in y_train]

    # Convert indexes to binary class matrix (for use with categorical crossentropy loss)
    y_train = to_categorical(y_train, num_classes)
    print('Train labels shape:', y_train.shape)

    print('Building a model with 1 hidden layer...')
    model = Sequential()  # Basic type of TensorFlow Keras models: a sequential stack of layers

    # We now start adding layers.
    # 'Dense' here means 'fully-connected':
    model.add(Dense(args.hidden_dim, input_shape=(embedding.vector_size,), activation='relu',
                    name="Input"))  # Specifying the input shape is important!

    model.add(Dropout(0.1))  # We will use dropout after the first hidden layer

    model.add(Dense(num_classes, activation='softmax', name='Output'))  # Output layer

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print out the model architecture
    print(model.summary())

    # We will monitor the dynamics of accuracy during training
    # If it stops improving, we will stop training.
    if OVERFIT:
        earlystopping = EarlyStopping(monitor='acc', min_delta=0.0001, patience=3, verbose=1,
                                      mode='max')
    else:
        earlystopping = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=4, verbose=1,
                                      mode='max')

    # what part of the training data will be used as a validation dataset:
    val_split = 1 - args.split

    # Train the compiled model on the training data
    # See more at https://keras.io/models/sequential/#sequential-model-methods
    start = time.time()
    if BALANCED:
        history = model.fit(x_train, y_train, epochs=args.epochs, verbose=1,
                            validation_split=val_split, batch_size=args.batch_size,
                            callbacks=[earlystopping], class_weight=class_weights)
    else:
        history = model.fit(x_train, y_train, epochs=args.epochs, verbose=1,
                            validation_split=val_split, batch_size=args.batch_size,
                            callbacks=[earlystopping])

    end = time.time()
    training_time = int(end - start)
    print('Training took:', training_time, 'seconds')

    val_nr = int(len(x_train) * val_split)  # Number of instances used for validation
    x_val = x_train[-val_nr:, :]
    y_val = y_train[-val_nr:, :]

    score = model.evaluate(x_val, y_val, verbose=2)
    print('Validation loss:', round(score[0], 4))
    print('Validation accuracy:', round(score[1], 4))

    # We calculate F1 score on the dev set:
    predictions = model.predict(x_val)

    # Convert predictions from integers back to text labels:
    y_test_real = [classes[int(np.argmax(pred))] for pred in y_val]
    predictions = [classes[int(np.argmax(pred))] for pred in predictions]
    fscore = precision_recall_fscore_support(y_test_real, predictions, average='macro')[2]
    print('Macro F1 on the dev set:', round(fscore, 2))

    # Saving the model to file
    OUT_RES = '%sclassifier/' % OUT
    os.makedirs(OUT_RES, exist_ok=True)

    model_filename = OUT_RES + run_name + '.h5'
    model.save(model_filename)
    with open(OUT_RES + run_name + '_classes.json', 'w') as f:
        f.write(json.dumps(classes, ensure_ascii=False, indent=4))
    print('Model saved to', model_filename)
    print('Classes saved to', run_name + '_classes.json')
    tf.keras.backend.clear_session()
    print('===========================')
