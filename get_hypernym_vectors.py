#! python3
# coding: utf-8

from hyper_import_functions import load_embeddings, predict, get_vector
from argparse import ArgumentParser
import os, sys
from smart_open import open
import pickle
import numpy as np

# python3 code/hypohyper/get_hypernym_vectors.py --provided_test data/hypohyper/public_test/nouns_public.tsv
# --projection proj/hypohyper/output/araneum_projection.pickle.gz --emb_name araneum
# # --emb_path resources/emb/araneum_upos_skipgram_300_2_2018.vec.gz --out proj/hypohyper/output/

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--provided_test', default='hypohyper/input/data/training_nouns.tsv', help='a list of hyponyms to attach to ruWordNet taxonomy', type=os.path.abspath)
    parser.add_argument('--projection', default='output/araneum_projection.pickle.gz',
                        help='.pickle.gz, the transformation matrix leanrt in the previous step', type=os.path.abspath)
    parser.add_argument('--emb_name', default='araneum',
                        help="arbitrary name of the embedding for output formatting purposes: rdt, araneum, cc, other")
    parser.add_argument('--emb_path', default='input/resources/araneum_upos_skipgram_300_2_2018.vec.gz',
                        help="path to embeddings file")
    parser.add_argument('--nr', type=int, default=10, help='Number of candidates')
    parser.add_argument('--out', default='output/', help="path to folder where to store hypernym projections", type=os.path.abspath)
    parser.add_argument('--tags', dest='tags', action='store_true', help="POS tags in embeddings?")
    
    parser.set_defaults(tags=True)
    args = parser.parse_args()
    

    datafile = args.provided_test
    modelfile = args.emb_path
    test_hyponyms = [i.strip().lower()+'_NOUN' for i in open(datafile, 'r').readlines()]

    print('Current embedding model:', modelfile, file=sys.stderr)
    model = load_embeddings(modelfile)

    if args.tags == True:
        vec = get_vector('дом_NOUN', emb=model)
        print('%%%%%%%%%%%%%%%%%%%%%', len(vec))
    if args.tags == False:
        vec = get_vector('дом', emb=model)
        print('=====================', len(vec))

    pickle_file = open(args.projection, 'rb')
    pickle_data = pickle.load(pickle_file)

    threshold = pickle_data['threshold']
    print('Using threshold:', threshold, file=sys.stderr)
    projection = pickle_data['projection']

    predicted = {}  # Predicted dictionary of hypernyms corresponding to each hyponym

    print('We will make predictions for %d hyponyms' % len(test_hyponyms), file=sys.stderr)

    print('Making predictions...', file=sys.stderr)
    counter = 0
    test_in_voc = []
    oov_in_test = []
    hyper_collector = []
    for hyponym in test_hyponyms:
        # print(hyponym)
        try:
            candidates, predicted_vector = predict(hyponym, model, projection, topn=args.nr)
            test_in_voc.append(hyponym[:-5].upper()) ## formatting to meet ruWordNet conventions
            hyper_collector.append(predicted_vector)
        except:
            candidates = None
            oov_in_test.append(hyponym)
        
        if candidates:
            if threshold:
                # Filtering stage
                # We allow only candidates which are not further from the projection
                # than one sigma from the average similarity in the true set
                rejected = [c for c in candidates if c[1] < threshold]
                candidates = [c for c in candidates if c[1] >= threshold]
            else:
                rejected = []
            # End filtering stage
        
            ## collecting possible hypernyms from the embeddings vocabulary
            candidates = [i[0] for i in candidates if i[0] != hyponym]
            predicted[hyponym] = candidates

        if counter % 100 == 0:
            print('%d hyponyms processed out of %d total' % (counter, len(test_hyponyms )),
                  file=sys.stderr)
            # Want to see predictions in real time?
            # print(hyponym, '\t', candidates)
        counter += 1
    print('Number of test words not in embeddings: %d (%d%%)' % (len(oov_in_test), len(oov_in_test)/len(test_hyponyms)*100))
    print('OOV in test\n%s' % oov_in_test)
    print('Saving the predicted vectors and the list of preprocessed %d test hyponyms that are found in vectors' % (len(test_hyponyms)-len(oov_in_test)))

    np.save('%s/%s_hyper_collector.npy' % (args.out, args.emb_name), hyper_collector)
    np.save('%s/%s_preprocessed_test.npy' % (args.out, args.emb_name), test_in_voc)
    if len(hyper_collector) == len(test_in_voc):
        print('Sanity test: passed')
