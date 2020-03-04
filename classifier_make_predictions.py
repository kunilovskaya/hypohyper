#! python
# coding: utf-8

import json
from argparse import ArgumentParser
import numpy as np
from smart_open import open
from tensorflow.keras.models import load_model

from hyper_imports import load_embeddings

if __name__ == '__main__':
    # add command line arguments
    # this is probably the easiest way to store args for downstream
    parser = ArgumentParser()
    parser.add_argument('--path', required=True, help="Path to the testing set", action='store')
    parser.add_argument('--w2v', required=True, help="Path to the embeddings")
    parser.add_argument('--run_name', default='test',
                        help="Human-readable name of the run. "
                             "Will be used to find the model file and classes file")
    parser.add_argument('--nsynsets', action='store', type=int, default=10,
                        help='How many synsets to keep at test time')
    parser.add_argument('--pos', default='NOUN', help="PoS tag to append to words")
    args = parser.parse_args()

    RUN_NAME = args.run_name
    EMB_MODEL = args.w2v
    TESTFILE = args.path

    embedding = load_embeddings(EMB_MODEL)

    print('Loading the model...')
    with open(RUN_NAME + '_classes.json', 'r') as f:
        classes = json.load(f)
    model = load_model(RUN_NAME + '.h5')
    print(model.summary())

    test_words = json.load(open(TESTFILE, 'r'))
    test_words = [w.lower() + '_' + args.pos for w in test_words]
    print('We have %d test words' % len(test_words))
    oov_counter = 0
    test_instances = []
    for word in test_words:
        if word in embedding:
            test_instances.append(embedding[word])
        else:
            test_instances.append(np.zeros(embedding.vector_size))
            oov_counter += 1
    print('OOV:', oov_counter)
    test_instances = np.array(test_instances)
    predictions = model.predict(test_instances)
    real_predictions = [list((-pred).argsort()[:args.nsynsets]) for pred in predictions]
    real_predictions = [[classes[s] for s in pred] for pred in real_predictions]
    out = {}
    for word, pred in zip(test_words, real_predictions):
        out[word.split('_')[0].upper()] = pred
    outname = '%s_%d_synsets_predictions.json' % (RUN_NAME, args.nsynsets)
    with open(outname, 'w') as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=4))
    print('Predictions saved to', outname)
