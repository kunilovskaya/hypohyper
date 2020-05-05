#! python
# coding: utf-8
import os
import json
from argparse import ArgumentParser
import numpy as np
from smart_open import open
from tensorflow.keras.models import load_model
import zipfile
from trials_errors.hyper_imports import load_embeddings

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv',
                        type=os.path.abspath, action='store')
    parser.add_argument('--w2v', default='204.zip', help="Path to the embeddings")
    parser.add_argument('--run_name', default='my_model',
                        help="Human-readable name of the run. "
                             "Will be used to find the model file and classes file")
    parser.add_argument('--nsynsets', action='store', type=int, default=10,
                        help='How many synsets to keep at test time')
    parser.add_argument('--POS', action='store', default='NOUN',
                        help='What part of speech do we work with?')
    args = parser.parse_args()

    RUN_NAME = args.run_name
    EMB_MODEL = args.w2v
    TESTFILE = args.test

    print('Loading the model...')
    with open(RUN_NAME + '_classes.json', 'r') as f:
        classes = json.load(f)
    model = load_model(RUN_NAME + '.h5')
    print(model.summary())

    print('Loading word embeddings...')
    embedding = load_embeddings(EMB_MODEL)

    test_words = open(TESTFILE, 'r').readlines()
    test_words = [w.strip().lower() + '_' + args.POS for w in test_words]
    print('We have %d test words, for example:' % len(test_words))
    print(test_words[:5])
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
        
    outname = 'pred.json'

    with open(outname, 'w') as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=4))
    print('Predictions saved to', outname)

    predictions = json.load(open('pred.json', 'r'))
    
    submission = outname.replace('.json', '.tsv')

    with open(submission, 'w') as f:
        for word in predictions:
            for synset in predictions[word]:
                f.write('\t'.join([word, synset, 'whatever']) + '\n')
    print('Inspect submission:', submission)
    # upload this archive to the site
    archive_name = 'pred.zip'
    with zipfile.ZipFile(archive_name, 'w') as file:
        file.write(submission, 'pred.tsv')
    print('Submit to codalab:', submission.replace('.tsv', '.zip'))

