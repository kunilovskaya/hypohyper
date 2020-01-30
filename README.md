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
(0) download the embeddings to the input/resources folder

**suggested options (pre-selected based on coverage and/or specialisation): 69913 and 84705 pairs respectively out of 94115 single word pairs)**

* (default and recommended) codename: [araneum](https://rusvectores.org/static/models/rusvectores4/araneum/araneum_upos_skipgram_300_2_2018.vec.gz) (**192 MB**, tags=True, binary=False); 
* codename: [rdt](http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz500-w10-cb0-it3-min5.w2v) (13 GiB, tags=False, binary=True, embeddings from Russian Distributional Thesaurus, limit=3500000);
* codename: [cc](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz) (1.3 GB, tags=False, binary=False, fasttext)

NB! all the running comands below assume that you accept the default settings, such as (inter alia)

* vectors trained on UD-tagged araneum
* in train-test MWE are joined with :: by default and all items are lowercased
* taking into account only single_wd members of synsets (and thus ignoring 19.3% of synsets)

(1) to get compressed train and 0.2 test files (.tsv.gz) with all hypo-hyper wordpairs from train reduced to only those found in the given embedding file (of 431937 wordpairs available, the best coverage of 84705 is see in RDT, araneum is second best with 69913)

```{r, engine='python3', count_lines}
python3 code/hypohyper/format_train.py 
```

(2) to get the .pickle.gz, which has {threshold: transforms} dict for subsequent internal evaluation against the embeddings vocabulary (see test_projection.py)

```
python3 code/hypohyper/learn_projection.py
```

(3) run internal evaluation (what are your MAP scores? MRR is the same, except it does not take into the account duplicates among predictions)

```
python3 code/hypohyper/test_projection.py
```

(4) to get a compressed file (.npz) with the predicted hypernym vectors, run 

```
python3 code/hypohyper/get_hypernym_vectors.py
```
### Produce the output file, given a list of hypernym vectors for each test word
(4) format the output: it is done in two steps to avoid uploading the embeddings yet again

* run a script to represent all (single word or multi-word) senses of NOUN synsets in ruWordNet and save the compressed index and vectors (as parts of .npz). Mind that if you want to include vectors for main\_words in MWE, replace single\_wd with main\_wd in the --mode option

```
python3 code/hypohyper/vectorise_ruwordnet.py --mode single_wd
```

* finally, to measure the similarities and produce the formatted output, run 

```
python3 code/hypohyper/measure_sims.py
```



## FYI (some stats for Nouns): 
* ratio of unigram senses to all senses 48.64% (total nounal 76817); 
* ratio of synsets that have NO unigram representation 19.30%


