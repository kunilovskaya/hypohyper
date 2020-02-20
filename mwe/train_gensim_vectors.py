# python mwe/train_gensim_vectors.py -t /home/rgcl-dl/Data/hypohyper/mwe-corpus_araneum-rncwiki-news-rncP-pro.gz/ -fr w2v -m cbow -w 5 -minc 10 -iter 5 -s binary

import gensim
import logging, sys
### I am learning throu wrappers
from gensim.models import FastText
from gensim.models import Word2Vec
import multiprocessing
import argparse

import time

start = time.time()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# How many workers (CPU cores) to use during the training?
cores = multiprocessing.cpu_count()  # Use all cores we have access to
logger.info('Number of cores to use: %d' % cores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', '-t', help="Path to training corpus", required=True)
    parser.add_argument('--framework', '-fr', default='w2v', type=str, help="Choose framework", choices=['w2v', 'ft'])
    parser.add_argument('--model', '-m', default='cbow', help="Name of the model", choices=['skipgram', 'cbow'])
    parser.add_argument('--window', '-w', default=5, help="Number of context words to consider", type=int)
    parser.add_argument('--mincount', '-minc', default=10, type=int, help="Min freq for vocabulary items: for raw input 10, for lemmatized 50")
    parser.add_argument("-iter", default=5, help="Run more training iterations (default 5)", type=int)
    parser.add_argument('--save', '-s', default='binary', help="Name of the model", choices=['binary', 'gensim', 'text'])
    args = parser.parse_args()


corpus = args.train
if args.model == 'skipgram':
        sg = 1
elif args.model == 'cbow':
        sg = 0
framework = args.framework
window = args.window
mincount = args.mincount
save = args.save
iterations = args.iter

logger.info(corpus)


data = gensim.models.word2vec.LineSentence(corpus)

if framework == 'w2v':
        outfile = corpus.replace('.gz', '_w2v') + '_' + str(args.model) + '_' + str(window)+ '_300_hs'
        ## Initialize and train a :class:`~gensim.models.word2vec.Word2Vec` model
        model = Word2Vec(data, size=300, sg=sg, min_count=mincount, window=window,
                hs=0, negative=5, workers=cores, iter=iterations, seed=42)
        
        if args.save == 'gensim':
                model.save(outfile + '.model')
        elif args.save == 'binary':
                model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
        elif args.save == 'text':
                model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)
        
if framework == 'ft':
        ### I don't understand abt these formats
        outfile = corpus.replace('.gz','_ft')+'_'+str(args.model)+'_'+str(window)+ '_3-6_hs' + '.model'
        ## Train, use and evaluate word representations learned using FastText with subword info
        model = FastText(data, size=300, sg=sg, min_count=mincount, window=window, min_n=3, max_n=6,
                hs=0, negative=5, workers=cores, iter=iterations, seed=42)
        ## The model can be stored / loaded via its: meth:`~gensim.models.fasttext.FastText.save`
        # and :meth: `~gensim.models.fasttext.FastText.load` methods
        ## or
        # model = FastText.load(fname)
        model.save(outfile)

end = time.time()
training_time = int(end - start)
print('Training embeddings on %s took %.2f minites' % (corpus.split('/')[-1], training_time/60), file=sys.stderr)

