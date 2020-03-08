KuKuPl team's contribution to the shared task at Dialogue Evaluation 2020: [Taxonomy Enrichment for the Russian Language](https://competitions.codalab.org/competitions/22168)


## tasks and solutions (last updated Mar07, 2020)

### (1) Getting the hypernym vector for a test word

**(1.1) select the best performing vector model in the no-frills setting**

we tried 8 vector models listed here in the order of performance on the NOUN public test 
- w2v upos araneum **NOUN: 0.2470** for top-hyper OOV strategy
- w2v_rdt500
- 186_tayga-pos5
- 182_ruscwiki-pos
- ft_araneum_ft_no_OOV
- news_pos_0_5
- ft_ruscorp_ft_no_OOV
- ft_araneum_full
- dedicated news ft with extended MWE support (('набат_NOUN', 'колокольный::звон_NOUN'), ('этиловый::спирт_NOUN', 'спирт_NOUN'))

NB! all fastText vectors are learnt on lemmatised corpora
see Tables 1-3 below for models coverage and comparative results

**(1.2) devise a strategy to fight OOV in testset**
- get top ten most frequent hypernym synsets for OOV test items
- get FastTest rpresentations
- cooccurence and/or Hearst statistics

**(1.3) test alternative learning METHODs**
* try training on avaraged synset vectors (deworded)
* apply negative sampling (neg-hyp, neg-syn) 
(adoptation of Ustalov's [hyperstar2017](https://arxiv.org/pdf/1707.03903) or make use of [PatternSims](https://github.com/cental/patternsim))
* use cooccurrence/Hearst statistics (corpus-informed25)
* learn a classifier to directly predict the hypernym synset id for testwords, train on hyponym vectors and the assigned hypernym ids
* increase the training by converting MWE from ruWordNet found in a huge corpus into single tokens (ex. wild_ADJ::bird_NOUN) and learning dedicated vectors for them

### ~~(2) Finding most-similar synset in ruWordNet~~ -- Get the first id for the words, found most similar to the predicted hypernym vector

**(2.1) decide how to approach MWE in ruWordNet (1/4 of synsets don't have single word lemmas, which renders them unavailable for hypernym synset id search)**
* limit the ruWordNet lemma search space to single-word items only
* for the synsets that do not have single word representations use main words of their MW lemmas
* instead of using word vectors, get averaged synset vectors as hyponym candidates~~

**(2.2) simple tricks**
* exclude duplicates and selves in the output

### House-keeping
**decide how representative is the Codalab public test and set up internal evaluation for error analysis** 
* ~~produce a test including only monosemantic items as is the case with the Codalab testset according to the orgs~~ BAD IDEA
* incorporate the train-dev-test split provided by the org Feb11

=====================================

## Outstanding tasks (Feb04):
- [X] ditch intrinsic evaluation and rerun on all data
- [X] analyse OOV in private tests: how important is anti-OOV strategy?
- [X] **FAILED**: produce a raw static train-test split so that test includes only monosemantic hyponym synsets, 
i.e. synsets that have only one hypernym synset; **Next step** fall back to random train-test split and see whether Codalab res will be reproducible
- [X] average synset vectors at train and hyponym-synset-detection times and for getting most_similar
- [X] exclude same word as hypernym; **contrary to expectations, I don't get same-name candidates in testset**
- [X] factor in cooccurence stats or 
- [X] Hearst patterns based on the news corpus
- [X] **FAILED: unusable for data with no gound truth available**; cluster input following Fu et al. (2014) and Ustalov (2017)'s suggestions 
- [X] **Results are lower than for the naive approach**: add negative sampling (based on hyponyms, synonyms) following Ustalov (2017)'s suggestions
- [ ] choose wiser: rank synsets and get the most high ranking ones
- [X] retain sublists in the reference to take into account "компоненты связности"
- [X] turn 36 K MWE from ruWordNet into tokens and learn dedicated vectors for them -- increases the training data 4 times, renders all ids accessible in producing final results
- [ ] ~~use coocurrence for OOV in test~~ -- solved by the dedicated vectors learned on 1 billion sentence corpus
- [ ] produce vectors for tokenized VERB MWE and run the classifier on VERBS
- [ ] write up the paper (till March 15, 2020)

## RUNNING the code 
(available for the main pipeline in full: 
- projections with negative sampling need to be learnt separately; 
- the 3d-place-winning classifier has separate code)

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
python3 get_static_test.py 
```

(1) to get compressed train and 0.2 test files (.tsv.gz) with all hypo-hyper wordpairs from train reduced to only those found in the given embedding file (of 431937 wordpairs available, the best coverage of 84705 is see in RDT, araneum is second best with 69913)

```
python3 format_data.py 
```

(2) to get the .pickle.gz, which has {threshold: transforms} dict for subsequent internal evaluation against the embeddings vocabulary (see test_projection.py)

```
python3 learn_projection.py
```

(3) to get a compressed file (.npz) with the predicted hypernym vectors, run 

```
python3 get_hyper_vectors.py
```

##### Produce the output file, given a list of hypernym vectors for test words

(5) this will produce the formmated output expected at the competition side

* finally, represent all (single word or multi-word) senses of NOUN synsets in ruWordNet, measure the similarities and produce the formatted output by running

```
python3 predict_synsets.py
```
(7-8)
if you are using random or intrinsic TEST option you will train on 90% of the data; you can produce the golden test set
to internally evaluate the performance of our algo

```
python3 intrinsic_evaluate.py
```

==================================================

## SOME RESULTS
### Comparative Results for various embeddings and basic setups (admittedly without taking into account the polysemy caveate in evaluation)

Table 1. Coverage (431937 noun and 233401 verb pairs) and OOV in testsets (762 nouns and 175 verbs)

vectors           | coverage_N | testOOV_N | coverage_V | testOOV_V |
:-----------------|---------:  |--------:  |---------:  |--------:  |
w2v_rdt500        | 84705      |  31(4%)   | 98609      | 6(3%)     |
w2v_pos_araneum   | 69916      |  32(4%)   | 41385      | ~~75(42%)~~   |
ft_araneum(no OOV)| 73384      |  22(2%)   | 41664      | ~~75(42%)~~   |
~~ft_ruscorp(no OOV)~~| 61213      |~~214(28%)~~   | 85189      | ~~30(17%)~~   |
~~w2v_pos_news~~      | ~~56761~~      |~~152(19%)~~  | 71915      | ~~55(31%)~~  |
ft_news           | 67124      |  38(4%)   | 78654      | 22(12%)      |
182_ruswiki_upos  | 55448      |  212(27%) |            |              |
~~186-tayga-fpos5~~ |  59182   |  ~~225(29%)~~ |            |              |

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


**Table 3. OOV in private test (nouns and verbs)**

vectors             |  NOUNS  |  VERBS   |
   :----------------|--------:|---------:|
 w2v_rdt500         |  55(3%) |   11(3%) |
 w2v_upos_araneum   |  56(3%) | ~~142(40%)~~ |
 ft_araneum         |  46(3%) | ~~142(40%)~~ |
 ~~ft_ruscorp~~     | 431(28%)| 69(19%)  |
 ~~w2v_pos_news~~   | ~~335(21%)~~| ~~116(33%)~~ |
 ft_news_cbow       |   88(5%)| 38(10%)  |

================================================================

Official results of the competition:
[NOUNS](https://competitions.codalab.org/competitions/22168#results)
A neural classifier learnt on hyponyms represented by dedicated vectors (i.e. on all training data available) beats projection-based approaches
 
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




