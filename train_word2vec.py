#! python
# coding: utf-8

import gensim
import logging
import multiprocessing
import argparse

# This script trains a word2vec word embedding models using Gensim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--corpus', help='Path to a training corpus', required=True)
    arg('--cores', default=False, help='Limit on the number of cores to use')
    arg('--sg', default=1, type=int, help='Use Skipgram (1) or CBOW (0)')
    arg('--window', default=5, type=int, help='Size of context window')
    arg('--mincount', default=10, type=int, help='Minimal frequency')
    arg('--vocab', default=100000, type=int, help='Max vocabulary size')
    args = parser.parse_args()

    # Setting up logging:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # This will be our training corpus to infer word embeddings from.
    # Most probably, a gzipped text file, one doc/sentence per line:
    corpus = args.corpus
    logger.info(corpus)

    data = gensim.models.word2vec.LineSentence(corpus)  # Iterator over lines of the corpus

    # How many workers (CPU cores) to use during the training?
    if args.cores:
        cores = int(args.cores)  # Use the number of cores we are told to use (in a SLURM file, for example)
    else:
        cores = multiprocessing.cpu_count()  # Use all cores we have access to
    logger.info('Number of cores to use: %d' % cores)

    # Setting up training hyperparameters:
    skipgram = args.sg  # Use Skipgram (1) or CBOW (0) algorithm?
    window = args.window  # Size of the symmetric context window (for example, 2 words to the right and 2 words to the left).
    vocabsize = args.vocab  # How many words types we want to be considered (sorted by frequency)?
    mincount = args.mincount
    vectorsize = 300  # Dimensionality of the resulting word embeddings.
    iterations = 3  # For how many epochs to train a model (how many passes over corpus)?

    # Start actual training!
    # Subsampling ('sample' parameter) is used to stochastically downplay the influence of very frequent words.
    # Since our corpus is most probably already filtered for stop words (functional parts of speech),
    # we do not need subsampling and set it to zero.
    model = gensim.models.Word2Vec(data, size=vectorsize, window=window, workers=cores, sg=skipgram, seed=42, max_final_vocab=vocabsize, iter=iterations, sample=0, min_count=mincount)
    # model = gensim.models.Word2Vec(data, size=vectorsize, window=window, workers=cores, sg=skipgram, seed=42, max_final_vocab=vocabsize, iter=iterations, min_count=mincount)

    # Saving the resulting model to a file
    # model = model
    filename = corpus.replace('.txt.gz','')+'_'+str(skipgram)+'_'+str(window)+'.model'
    logger.info(filename)
    model.wv.save(filename)
