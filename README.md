KuKuPl team's contribution to the shared task at Dialogue Evaluation 2020: [Taxonomy Enrichment for the Russian Language](https://competitions.codalab.org/competitions/22168)

## RESULTS (last updated Feb04, 2020)

### Discussion:
* (unexpectedly) intrinsic evaluation returns lower results for RDT vectors of size 500
* intrinsic evaluation on 0.2 test is only arbitrary related to evaluation on the public test, due to the dynamic nature of our testset and uncontrolled polysemy of hyponyms (which was avoided in the public (real) test) 
* we tried 6 vector models listed here in the order of performance on the NOUN public test (all fastText vectors are learnt on lemmatised corpora)
- w2v upos araneum **NOUN: 0.2540**
- w2v_rdt500
- ft_araneum_ft_no_OOV
- ft_ruscorp_ft_no_OOV
- ft_araneum_full
- dedicated news ft with extended MWE support (('набат_NOUN', 'колокольный::звон_NOUN'), ('этиловый::спирт_NOUN', 'спирт_NOUN'))
* the best performing combination of parameters for getting synsets most similar to hypernym projection and for resolving OOV problem in testset: limit the ruWordNet lemma search space by single-word items only and get top ten most frequent hypernym synsets for OOV test items.
* w2v works better than ft on the same corpus (we are discarding all OOV training pairs; the performance on the full 4 times bigger train represented with fastText is considerably lower)


### Comparative Results for various setups

Table 1. Coverage (431937 noun and 233401 verb pairs) and OOV in testsets (762 nouns and 175 verbs)

vectors           | coverage_N | testOOV_N | coverage_V | testOOV_V |
:-----------------|---------:  |--------:  |---------:  |--------:  |
w2v_rdt500        | 84705      |  31(4%)   | 98609      | 6(3%)     |
w2v_pos_araneum   | 69916      |  32(4%)   | 41385      | ~~75(42%)~~   |
ft_araneum(no OOV)| 73384      |  22(2%)   | 41664      | ~~75(42%)~~   |
~~ft_ruscorp(no OOV)~~| 61213      |~~214(28%)~~   | 85189      | ~~30(17%)~~   |
~~w2v_pos_news~~      | ~~56761~~      |~~152(19%)~~  | 71915      | ~~55(31%)~~  |
ft_news           | 67124      |  38(4%)   | 78654      | 22(12%)      |

* Number of MWE (ft_news):      697
* Number of MWE (w2v_pos_news): 32

### NOUNS                   
**FYI(Feb03): baseline MAP=0.1405; best competitor MAP=0.4282**  
 
Table 2A. NOUN: For some models we report results (publicMAP values for models train **on ALL train**) on the combination of approaches to (a) ruWordNet vectorisation and to (b) test OOV elimination 
(switch to raw text to see the table)

 vectors            |single+ft_vectors|single+top_hyper|main+ft_vectors|main+top_hyper|
  :---------------- |------------:|-----------:|------------:|-----------:| 
 w2v_rdt500         |   N/A       |   0.2163** |     N/A     |   --       |
 w2v_pos_araneum    | **0.2540**  |   0.2484   |   0.1617    |   0.1600   |
 ft_araneum(full)   |    --       |    --      |   0.0756*   |    --      |
 ft_araneum(no_OOV) |    0.1406   |   0.1349   |   0.1381    |   0.0837   |
 ft_ruscorp(no_OOV) |    0.0685*  |   0.0683   |      --     |      --    |
 w2v_pos_news_cbow  |    0.1307   |   0.1279   |   --        |   0.0763   |
 w2v_pos_news_sk    |    --       |   0.1056   |   0.0633    |   --       |
 ft_news_cbow       |   0.0338    |   --       |   --        |   --       |
 ft_news_sk         |   --        |   0.0766   |   --        |   --       |

(*) results for 100 most_similar words in the embeddings; the default is 500
(**) result for 1000 most similar (output length is 7042 instead of 7620)

* **single_wd:** When selecting the hypernym synset, use only single-word lemmas of the synset (and ignore  16% of 76817 senses (for w2v_upos_araneum) that have not vectors in embeddings (OOV)); **31205 vectorised senses from 21188 synsets**
* **main_wd:** Choosing from all synsets, including those with no single_word representations (by using main words of the MWE); **69951 vectorised senses from 27536 synsets**
                                    
### VERBS
**FYI: baseline MAP=0.0712; best competitor MAP=0.2756**

    >> ex. изъедать_VERB    ['попортить_VERB', 'изгрызать_VERB', 'сжирать_VERB', 'испоганивать_VERB', 'испортить_VERB']

**VERBS: Intrinsic evaluation on 0.2 test**
- w2v_upos_araneum: MAP: 0.0634, MRR: 0.1091

Table 2B. VERBS: Different approaches to process ruWordNet and OOV in test
Evaluation is stuck due to Codalab problems

  vectors      | single+ft_vectors|single+top_hyper|main+ft_vectors|main+top_hyper|  
   :----------------|------------:|-----------:|------------:|-----------:|
 w2v_rdt500         |      N/A    | **0.0850** |      N/A    |            |
 w2v_upos_araneum   |    --       |   0.0599   |   --        |  --        |
 ft_araneum(no_OOV) |    0.0599   |   --       |   --        |  --        |
 ft_ruscorp(no_OOV) |             |  0.0319     |   --        |  --        |
 w2v_pos_news_cbow  |    0.0174   |   --   |   --        |   --       |
 w2v_pos_news_sk    |    --       |   0.0361   |   0.0146    |   --       |
 ft_news_cbow       |   0.0006    |   --       |   --        |   --       |
 ft_news_sk         |   0.0174   |   --   |   --        |   --       |

=========================================================================

**Table 3. OOV in private test (nouns and verbs)**

vectors             |  NOUNS  |  VERBS   |
   :----------------|--------:|---------:|
 w2v_rdt500         |  55(3%) |   11(3%) |
 w2v_upos_araneum   |  56(3%) | ~~142(40%)~~ |
 ft_araneum         |  46(3%) | ~~142(40%)~~ |
 ~~ft_ruscorp~~     | 431(28%)| 69(19%)  |
 ~~w2v_pos_news~~   | ~~335(21%)~~| ~~116(33%)~~ |
 ft_news_cbow       |   88(5%)| 38(10%)  |

## The task breakdown

(1) produce/format the training data: hyponym---hypernym pairs from the training set provided 
* get all possible hypo-hypernym pairs row-wise from TEXT---PARENT\_TEXTS columns (get 431937 pairs, see provided [training\_data](https://github.com/dialogue-evaluation/taxonomy-enrichment/blob/master/data/training_data/training_nouns.tsv) )
* glue all MWE with :: (посевная::кампания_PROPN, научная::организация_PROPN); >90% of them  are filtered out anyway even with fasttext (if skip_oov=True)
* filter out embedding's OOV (mostly MWE): this reduces the train to less than a quarter of the original number of pairs

(2) learn a transformation matrix to go from a hyponym vector to a hypernym vector
* even for fasttext skip OOV (done in step 1, actually)
* train/test split (test_size=.2)
* args.lmbd = 0.0
* _TODO problem_: polysemy in train is pervasive: the length of 0.2 test is 13984 wordpairs, 
however, there are only 7274 unique hyponyms and 3622 cases of duplicates on the hyponym side of the pairs from the training data.

(3) detect the synsets that are most similar to the predicted hypernym vector
* represente synsets semantics or ruWordNet items (esp. MWE); choose mode: how to represent MWE lemmas in ruWordNet synset lemmas (one option is to include vectors for main components of MWE only if this synset has no single_word representation)
* decide how to get results for OOV in test: either use 10 synsets that are most frequent hypernyms in ruWordNet (OOV\_STRATEGY == 'top\_hyper') or use fasttext to produce vectors for them from parts (OOV\_STRATEGY = 'ft\_vector'). See stats on this in the Results tables.

===============================================================
## Outstanding tasks (Feb04):
- [x] ditch intrinsic evaluation and rerun on all data
- [X] analyse OOV in private tests: how important is anti-OOV strategy?
- [X] produce a raw static train-test split so that test includes only monosemantic hyponym synsets, i.e. synsets that have only one hypernym synset
- [ ] average synset vectors at train time and for getting most_similar
- [ ] factor in cooccurence stats or patterns based on the news corpus
- [X] cluster input (inapplicable for data with no gound truth available)
- [ ] add negative sampling (based on hyponyms, synonyms)
- [ ] choose wiser: rank synsets and get the most high ranking ones

================================================================

### Getting the resources and setting up the parameters
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
will run the pipeline of five scripts, create foders, save and load interim output and print the results of each step, provided that the preparatory steps i and ii are taken

#### Step-by-step

(0) make a static train-test split on raw data regardless of te embedding used, for comparability reasons

```
python3 get_intrinsic_test.py 
```

(1) to get compressed train and 0.2 test files (.tsv.gz) with all hypo-hyper wordpairs from train reduced to only those found in the given embedding file (of 431937 wordpairs available, the best coverage of 84705 is see in RDT, araneum is second best with 69913)

```
python3 format_data.py 
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
##### Produce the output file, given a list of hypernym vectors for test words
(5) format the output: it is done in two steps to avoid uploading the embeddings yet again

* finally, represent all (single word or multi-word) senses of NOUN synsets in ruWordNet, measure the similarities and produce the formatted output by running

```
python3 measure_sims.py
```

========================================================================

## FYI (some stats for Nouns): 
* ratio of unigram senses to all senses 48.64% (total nounal 76817); 
* ratio of synsets that have NO unigram representation 19.30%

>> Examples of OOV in testset: OOV in test

'барьерист', 'букраннер', 'вэйп', 'гольмий', 'городошник', 'градобитие', 'дефлимпиада', 'дресс-код', 'зоман', \
>'краболов', 'лжеминирование', 'мукоед', 'начштаб', 'недоносительство', 'неизбрание', 'папамобиль', 'прет-а-порте', \
>'троеборец', 'химлаборатория', 'черлидерша', 'чирлидерша', 'элефантиаз'

> редевелопмент, овербукинг, каршеринг, чирлидерша

## Ideas
**possible strategies**

* represent training pairs with fasttext (to address out-of-embeddings issues esp for MWE), train a binary classifier to predict whether hypernymy obtains between any given pair of words; for every test word build all possible wordpairs with all names of senses, predict hypernymy, replace words to synsets_ids, set ids, get top 10;
* learn a transformation matrix to go from a hyponym embedding to its hypernym vector; predict a hypernym vector for each (single word) test items; represent each sense/synset with a vector, return 10 pairs with the highest cosine similarity 
* failed to adopt Ustalov's [hyperstar2017](https://arxiv.org/pdf/1707.03903) or make use of [PatternSims](https://github.com/cental/patternsim)

**work in progress**

DONE: Can we reduce ruWordNet items to single word entries only to identify hypernym synsets ids? Are there many synsets that are lemmatized only via MWE?
TODO: get a generalized/averaged vector for each synset


