#! python
# coding: utf-8
import os
import json
from argparse import ArgumentParser
import numpy as np
from smart_open import open
from tensorflow.keras.models import load_model
from configs import OUT, POS, TEST, VECTORS, EMB_PATH, METHOD, FILTER_1, FILTER_2
import zipfile

from hyper_imports import load_embeddings

if __name__ == '__main__':
    # add command line arguments
    # this is probably the easiest way to store args for downstream
    parser = ArgumentParser()
    if TEST == 'codalab-pub':
        if POS == 'NOUN':
            parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath, action='store')
        if POS == 'VERB':
            parser.add_argument('--test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath, action='store')
    if TEST == 'codalab-pr':
        if POS == 'NOUN':
            parser.add_argument('--test', default='input/data/private_test/nouns_private.tsv', type=os.path.abspath, action='store')
        if POS == 'VERB':
            parser.add_argument('--test', default='input/data/private_test/verbs_private.tsv', type=os.path.abspath, action='store')
    if TEST == 'provided':
        parser.add_argument('--test', default='lists/%s_%s_WORDS.txt' % (POS, TEST), type=os.path.abspath)
            
    parser.add_argument('--w2v', default=EMB_PATH, help="Path to the embeddings")
    parser.add_argument('--run_name', default='notest_' + VECTORS + '_' + POS,
                        help="Human-readable name of the run. "
                             "Will be used to find the model file and classes file")
    parser.add_argument('--nsynsets', action='store', type=int, default=10,
                        help='How many synsets to keep at test time')
    args = parser.parse_args()

    RUN_NAME = args.run_name
    EMB_MODEL = args.w2v
    TESTFILE = args.test

    embedding = load_embeddings(EMB_MODEL)
    
    OUT_RES = '%sclassifier/' % OUT
    os.makedirs(OUT_RES, exist_ok=True)
    
    print('Loading the model...')
    with open(OUT_RES + RUN_NAME + '_classes.json', 'r') as f:
        classes = json.load(f)
    model = load_model(OUT_RES + RUN_NAME + '.h5')
    print(model.summary())

    test_words = open(TESTFILE, 'r').readlines()
    test_words = [w.strip().lower() + '_' + POS for w in test_words]
    print(test_words[:5])
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
        
    if 'codelab' in TEST:
        OUT_RES = '%sresults/' % OUT
        os.makedirs(OUT_RES, exist_ok=True)
        outname = '%s%s_%s_%s_%s_%s_pred.json' % (OUT_RES, POS, TEST, METHOD, FILTER_1, FILTER_2)
    elif TEST == 'provided':
        OUT_RES = '%sresults/org_split/' % OUT
        os.makedirs(OUT_RES, exist_ok=True)
        outname = '%s%s_%s_%s_%s_%s_pred.json' % (OUT_RES, POS, TEST, METHOD, FILTER_1, FILTER_2)
    else:
        OUT_RES = '%sresults/org_split/' % OUT
        os.makedirs(OUT_RES, exist_ok=True)
        outname = None
    
    with open(outname, 'w') as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=4))
    print('Predictions saved to', outname)

    predictions = json.load(open('%s%s_%s_%s_%s_%s_pred.json' % (OUT_RES, POS, TEST, METHOD, FILTER_1, FILTER_2), 'r'))
    
    submission = outname.replace('.json', '.tsv')

    with open(submission, 'w') as f:
        for word in predictions:
            for synset in predictions[word]:
                f.write('\t'.join([word, synset, 'whatever']) + '\n')
    print('Inspect submission:', submission)
    # upload this archive to the site
    archive_name = '%s_%s_%s_%s_%s_pred.zip' % (POS, TEST, METHOD, FILTER_1, FILTER_2)
    with zipfile.ZipFile(OUT_RES + archive_name, 'w') as file:
        file.write(submission,
                   '%s_%s_%s_%s_%s_pred.tsv' % (POS, TEST, METHOD, FILTER_1, FILTER_2))

    print('Submit to codalab:', submission.replace('.tsv', '.zip'))
    # os.remove(outfile)
