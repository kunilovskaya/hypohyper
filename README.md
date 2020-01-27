KuKuPl team's contribution to the shared task at Dialogue Evaluation 2020: [Taxonomy Enrichment for the Russian Language](https://competitions.codalab.org/competitions/22168)

# The task breakdown

(1) produce/format the training data: hyponym---hypernym pairs from the training set provided 
* get all possible hypo-hypernym pairs row-wise from TEXT---PARENT\_TEXTS columns (get 431937 pairs, see provided [training\_data](https://github.com/dialogue-evaluation/taxonomy-enrichment/blob/master/data/training_data/training_nouns.tsv) )
* filter out all MWE if using w2v embeddings (get 94115 pairs of single word items)
* filter out all pairs which have out of embeddings items

(2) train a model to get top 10 hypernym synsets

**possible strategies**

* represent training pairs with fasttext (to address out-of-embeddings issues esp for MWE), train a binary classifier to predict whether hypernymy obtains between any given pair of words; for every test word build all possible wordpairs with all names of senses, predict hypernymy, replace words to synsets_ids, set ids, get top 10;
* learn a transformation matrix to go from a hyponym embedding to its hypernym vector; predict a hypernym vector for each (single word) test items; represent each sense/synset with a vector, return 10 pairs with the highest cosine similarity 
* failed to adopt Ustalov's [hyperstar2017](https://arxiv.org/pdf/1707.03903) 

(3) approaches to representing synsets semantics or ruWordNet items (esp. MWE) for either classification task or for measuring similarity

**work in progress**

can we reduce ruWordNet items to single word entries only to identify hypernym synsets ids? Are there many synsets that are lemmatized only via MWE?

## How to run

### Training data preparation
(1) download the embeddings to the input/resources folder

**suggested options (pre-selected based on coverage and/or specialisation): 69913 and 84705 pairs respectively out of 94115 single word pairs)**

* codename: [araneum](https://rusvectores.org/static/models/araneum_upos_skipgram_600_2_2017.bin.gz) (419 MB, tags=True, binary=False); 
* codename: [rdt](http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz500-w10-cb0-it3-min5.w2v) (13 GiB, tags=False, binary=True, embeddings from Russian Distributional Thesaurus, limit=3500000);
 
(2) to run with the default araneum embeddings, from the main folder run _python3 format\_train.py_ (to change to RDT embeddings run _python3 format\_train.py --emb rdt_)

### Produce the output file, given a list of hypernym vectors for each test word

**work in progress**

