import numpy as np
from gensim import utils, models
import logging
import zipfile
import json

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def eval_func(batched_data, model2use):
    # This function uses a model to compute predictions on data coming in batches.
    # Then it calculates the accuracy of predictions with respect to the gold labels.
    correct = 0
    total = 0
    predicted = None
    gold_label = None

    # Iterating over all batches (can be 1 batch as well):
    for n, (data) in enumerate(batched_data):
        gold_label = data[-1]
        out = model2use(*data[:-1])
        predicted = out.max(dim=1)[1]
        correct += len((predicted == gold_label).nonzero())
        total += len(gold_label)
    accuracy = correct / total
    return accuracy, predicted, gold_label


def save_word2vec_format(fname, vocab, vectors, binary=False):
    """Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.
        Parameters
        ----------
        fname : str
            The file path used to save the vectors in
        vocab : dict
            The vocabulary of words with their ranks
        vectors : numpy.array
            The vectors to be stored
        binary : bool
            If True, the data wil be saved in binary word2vec format,
            else it will be saved in plain text.
        """
    if not (vocab or vectors):
        raise RuntimeError('no input')
    total_vec = len(vocab)
    vector_size = vectors.shape[1]
    print('storing %dx%d projection weights into %s' % (total_vec, vector_size, fname))
    assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        fout.write(utils.to_utf8('%s %s\n' % (total_vec, vector_size)))
        position = 0
        for element in sorted(vocab, key=lambda word: vocab[word]):
            row = vectors[position]
            if binary:
                row = row.astype(np.float32)
                fout.write(utils.to_utf8(element) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8('%s %s\n' % (element, ' '.join(repr(val) for val in row))))
            position += 1


def load_embeddings(embeddings_file):
    # Detect the model format by its extension:
    # Binary word2vec format:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        emb_model = models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True,
                                                             unicode_errors='replace')
    # Text word2vec format:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        emb_model = models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')
    # ZIP archive from the NLPL vector repository:
    elif embeddings_file.endswith('.zip'):
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open('meta.json')
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print('============')
            # Loading the model itself:
            stream = archive.open("model.bin")  # or model.txt, if you want to look at the model
            emb_model = models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:
        # Native Gensim format?
        emb_model = models.KeyedVectors.load(embeddings_file)
        # If you intend to train further: emb_model = models.Word2Vec.load(embeddings_file)

    emb_model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return emb_model

def safe_lookup(emb_model, word):
    try:
        return emb_model.vocab[word].index
    except KeyError:
        return 0
