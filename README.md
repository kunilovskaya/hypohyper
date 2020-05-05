KuKuPl team's contribution to the shared task at Dialogue Evaluation 2020:
[Taxonomy Enrichment for the Russian Language](https://competitions.codalab.org/competitions/22168)

The root folder has the code for the most successful approach -- a classifier trained on the customised word embeddings 
(which provide extended support for ruWordNet MWE) to predict synset ids.

## Training the classifier
To train a classifier, run

`python3 train_synset_classifier.py --path DATA --w2v EMBED --run_name TITLE`

where PATH_TO_DATA is a tab-separated training file, where the first column (with the *'word'* header) has words,
and the second column (with the *'synsets'* header) has lists of synset identifiers, one word per line, for example:

`урок_NOUN	["106723-N", "6718-N"]`

EMBED must be a file containing pre-trained word embeddings in pretty much any standard format
(plain text word2vec, binary word2vec, Gensim, NLPL zip archives...)
Make sure the words in the training data match the word forms in the embeddings!

The script will train the classifier and save it as an .h5 file
along with a JSON file containing an ordered list of synset identifiers. Their names are determined by the TITLE parameter.
Both files are required to make predictions with the trained classifier.

## Making predictions
Given a trained classifier predict hypernyms for words with:

`python3 make_predictions.py --test DATA --w2v EMBED --run_name TITLE`

where DATA is a plain text file with one word per line, and EMBED and TITLE have the same meaning

This will produce the `pred.json` file with 10 most probable hypernym synsets for each test word,
and the `pred.zip` file ready to be submitted to the shared task Codalab.

## Specially tailored embeddings we used
Download: http://vectors.nlpl.eu/repository/20/204.zip

Trained on lemmatised and UD-tagged:
- Araneum Russicum Maximum
- Russian Wikipedia
- Russian National Corpus (both the main and newspaper parts)
- Russian news corpus

In all the corpora, multi-word entities from ruWordNet were represented as one token. 
For example, *ящик_NOUN::из_ADP::картон_NOUN* and *давать_VERB::возможность_NOUN*


## Other stuff
The code for other, less successful approaches and their various settings, described in our paper (to be linked) and including
- projection learning,
- use of co-occurence statistics and
- Hearst patterns

 has been moved to *trial_error* folder.