KuKuPl team's contribution to the shared task at Dialogue Evaluation 2020: [Taxonomy Enrichment for the Russian Language](https://competitions.codalab.org/competitions/22168)

# RESULTS (last updated Feb03, 2020)

## SUMMARY and OBSERVATION:
* (unexpectedly) intrinsic evaluation returns lower results for RDT vectors of size 500

### ===== Comparative Results for various setups ======
### NOUNS for each anti-OOV-in-test strategy
FYI(Feb03): baseline MAP=0.1405; best competitor MAP=0.4282

Table 1. Intrinsic evaluation on 0.2 test (publicMAP values)

    vectors         |coverage (wdpairs)| intrMAP | intrMRR | pub/pr testOOV |
--------------------|-----------------:|--------:|--------:|---------------:|
     w2v_RDT        |      84705       |  0.0994 | 0.0926  |                |
 w2v_upos_araneum   |      69916       |  0.1168 | 0.1619  |  32(4%)/       |
 araneum_ft(full)   |     431937       |         |         |     --         |
--------------------|-----------------:|--------:|--------:|---------------:|
 araneum_ft(no_OOV) |      73384       |  0.1288 | 0.1047  |                |
 ruscorp_ft(no_OOV) |                  |         |         |                |
--------------------|-----------------:|--------:|--------:|---------------:|
 dedicated_news_ft  |                  |         |         |                |

Table 2. For some models we report results on the combination of approaches to (a) ruWordNet vectorisation and to (b) test OOV elimination

                    |       single_wd          |         main_wd          |
    vectors         | ft\_vectors | top\_hyper | ft\_vectors | top\_hyper |
--------------------|------------:|-----------:|------------:|-----------:|
     w2v_RDT        |             |            |             |            |
 w2v\_upos_araneum  |    0.2464   |   0.2467   |   0.1679    |  0.1682    |
--------------------|------------:|-----------:|------------:|-----------:|
 araneum\_ft(full)  |             |            |             |            |
 araneum_ft(no_OOV) |             |            |             |            |
 ruscorp_ft(no_OOV) |             |            |             |            |
--------------------|------------:|-----------:|------------:|-----------:|
 dedicated_news_ft  |             |            |             |            |

* **single_wd:** When selecting the hypernym synset, use only single-word lemmas of the synset (and ignore  16% of 76817 senses (for w2v_upos_araneum) that have not vectors in embeddings (OOV)); **31205 vectorised senses from 21188 synsets**
* **main_wd:** Choosing from all synsets, including those with no single_word representations (by using main words of the MWE); **69951 vectorised senses from 27536 synsets**



                                                               


### VERBS
FYI: baseline MAP=0.0712; best competitor MAP=0.2756

## The task breakdown

(1) produce/format the training data: hyponym---hypernym pairs from the training set provided 
* get all possible hypo-hypernym pairs row-wise from TEXT---PARENT\_TEXTS columns (get 431937 pairs, see provided [training\_data](https://github.com/dialogue-evaluation/taxonomy-enrichment/blob/master/data/training_data/training_nouns.tsv) )
* glue all MWE with :: (посевная::кампания_PROPN, научная::организация_PROPN); >90% of them  are filtered out anyway even with fasttext (if skip_oov=True)
* filter out embedding's OOV (mostly MWE): this reduces the train to less than a quarter of the original number of pairs

(2) learn a transformation matrix to go from a hyponym vector to a hypernym vector
* even for fasttext skip OOV (done in step 1, actually)
* train/test split (test_size=.2)
* args.lmbd = 0.0
* _TODO problem_: polysemy in train is pervasive: the length of 0.2 test is 13984 wordpairs, however, there are only 7274 unique hyponyms and 3622 cases of duplicates on the hyponym side of the pairs from the training data.

(3) detect the synsets that are most similar to the predicted hypernym vector
* represente synsets semantics or ruWordNet items (esp. MWE); choose mode: how to represent MWE lemmas in ruWordNet synset lemmas (one option is to include vectors for main components of MWE only if this synset has no single_word representation)
* decide how to get results for OOV in test: either use 10 synsets that are most frequent hypernyms in ruWordNet (OOV\_STRATEGY == 'top\_hyper') or use fasttext to produce vectors for them from parts (OOV\_STRATEGY = 'ft\_vector'). See stats on this in the Results tables.

### Getting the resource and setting up the parameters
(i) download the embeddings to the input/resources folder

**suggested options (pre-selected based on coverage and/or type)**

* (default and recommended) codename: [araneum](https://rusvectores.org/static/models/rusvectores4/araneum/araneum_upos_skipgram_300_2_2018.vec.gz) (**192 MB**, tags=True, binary=False); 
* codename: [rdt](http://panchenko.me/data/dsl-backup/w2v-ru/all.norm-sz500-w10-cb0-it3-min5.w2v) (13 GiB, tags=False, binary=True, embeddings from Russian Distributional Thesaurus, limit=3500000);
* codename: [cc](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz) (1.3 GB, tags=False, binary=False, fasttext)

(ii) set the paths and the parameters in the configs.py 

### Running the pipeline
NB! all the running comands below assume that you accept the default settings, such as (inter alia)

* vectors trained on UD-tagged araneum
* in train-test MWE are joined with :: by default and all items are lowercased
* taking into account only single_wd members of synsets (and thus ignoring 19.3% of synsets)

#### In one go

```
sh run_all.sh
```
will run the pipeline of six scripts and print the results of each step, provided that the preparatory steps i and ii are taken

#### Step-by-step

(1) to get compressed train and 0.2 test files (.tsv.gz) with all hypo-hyper wordpairs from train reduced to only those found in the given embedding file (of 431937 wordpairs available, the best coverage of 84705 is see in RDT, araneum is second best with 69913)

```{r, engine='python3', count_lines}
python3 format_train.py 
```

(2) to get the .pickle.gz, which has {threshold: transforms} dict for subsequent internal evaluation against the embeddings vocabulary (see test_projection.py)

```
python3 learn_projection.py
```

(3) run internal evaluation (what are your MAP scores? MRR is the same, except it does not take into the account duplicates among predictions)

```
python3 test_projection.py
```

(4) to get a compressed file (.npz) with the predicted hypernym vectors, run 

```
python3 get_hyper_vectors.py
```
### Produce the output file, given a list of hypernym vectors for each test word
(5) format the output: it is done in two steps to avoid uploading the embeddings yet again

* run a script to represent all (single word or multi-word) senses of NOUN synsets in ruWordNet and save the compressed index and vectors (as parts of .npz). Mind that if you want to include vectors for main\_words in MWE, replace single\_wd with main\_wd in the --mode option

```
python3 vectorise_ruwordnet.py --mode single_wd
```

* finally, to measure the similarities and produce the formatted output, run 

```
python3 measure_sims.py
```


## FYI (some stats for Nouns): 
* ratio of unigram senses to all senses 48.64% (total nounal 76817); 
* ratio of synsets that have NO unigram representation 19.30%

## Ideas
**possible strategies**

* represent training pairs with fasttext (to address out-of-embeddings issues esp for MWE), train a binary classifier to predict whether hypernymy obtains between any given pair of words; for every test word build all possible wordpairs with all names of senses, predict hypernymy, replace words to synsets_ids, set ids, get top 10;
* learn a transformation matrix to go from a hyponym embedding to its hypernym vector; predict a hypernym vector for each (single word) test items; represent each sense/synset with a vector, return 10 pairs with the highest cosine similarity 
* failed to adopt Ustalov's [hyperstar2017](https://arxiv.org/pdf/1707.03903) or make use of [PatternSims](https://github.com/cental/patternsim)

**work in progress**

DONE: Can we reduce ruWordNet items to single word entries only to identify hypernym synsets ids? Are there many synsets that are lemmatized only via MWE?
TODO: get a generalized/averaged vector for each synset


