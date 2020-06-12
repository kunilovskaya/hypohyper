RANDOM_SEED = 0
# codename for emb: 'mwe-pos-vectors', 'noun-verb_mwe_vectors', 'ft-araneum', w2v-rdt500, w2v-pos-ruscorpwiki, w2v-pos-araneum, ft-ruscorp, ft-araneum_full
# news-pos-sk, news-pos-cbow, ft-news-sk, ft-news-cbow, w2v-tayga-fpos5
VECTORS = 'mwe-pos-vectors' ## no full means that we filter out OOV at train time
POS = 'NOUN' # 'NOUN'
MWE = True ## always TRUE wouldn't hurt
TEST = 'provided' # , codalab-pub, codalab-pr, provided

## strategies to improve performance
METHOD = 'lemmas-neg-syn' # lemmas-neg-syn, lemmas-neg-syn, deworded, lemmas, classifier

# this this supposed to include vectors for main components of MWE only
# if this synset has no single_word representation or if MWE is found in vectors
MODE = 'single' # if you want to include vectors for main_words in MWE, replace single with main;

OOV_STRATEGY = 'top-hyper' ##'ft-vector', 'top-hyper'

# limit the number of similarities from vector model
if 'ft' not in VECTORS and POS == 'VERB':
    vecTOPN = 1500
elif 'rdt' in VECTORS:
    vecTOPN = 1000
else:
    vecTOPN = 100
## first number is how many of the hypers predicted by default to retain; second is how many top co-occuring lemmas to consider
## experimentally the best combination seems to be 25-25 or 15-25
if METHOD == 'classifier':
    FILTER_1 = 'raw'  # raw, disamb, comp, anno, corp-info25-15, corp-info50-30,corp-info25-15 corp-info15-25 hearst-info25-25 hearst-info50-25
    FILTER_2 = 'none'
else: # select filteres
    FILTER_1 = 'corp-info25-25' # raw, disamb, comp, anno, corp-info25-15, corp-info50-30,corp-info25-15 corp-info15-25 hearst-info25-25 hearst-info50-25
    FILTER_2 = 'none' #'kid', 'parent', none (for raw, disamb)

ALL_EMB = '/home/u2/resources/emb/'

if 'pos' in VECTORS or 'mwe' in VECTORS:
    TAGS = True
else:
    TAGS = False
    print('Looks like you dont have tags in embeddings?')
    
if VECTORS == 'w2v-pos-araneum':
    EMB_PATH = ALL_EMB + 'araneum_upos_skipgram_300_2_2018.bin'
elif VECTORS == 'mwe-pos-vectors':
    EMB_PATH = ALL_EMB + 'mwe_vectors/big_mwe_corpus_0_2.model'
elif VECTORS == 'noun-verb_mwe_vectors':
    EMB_PATH = ALL_EMB + 'noun-verb_mwe_vectors/nofunct_noun_verb_mwe_vectors_corpus_0_2.model'
elif VECTORS == 'w2v-pos-ruscorpwiki':
    EMB_PATH = ALL_EMB + '182_ruwikiruscorpora_upos_skipgram_300_2_2019/model.bin'
elif VECTORS == 'w2v-tayga-fpos5':
    EMB_PATH = ALL_EMB + '186_tayga-func_upos_skipgram_300_5_2019/model.bin'
elif 'ft-araneum' in VECTORS:
    EMB_PATH = ALL_EMB + 'araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
elif 'ft-ruscorp' in VECTORS:
    EMB_PATH = ALL_EMB + '181_ruscorpora_fasttext/model.model'
elif 'rdt' in VECTORS:
    EMB_PATH = ALL_EMB + 'all.norm-sz500-w10-cb0-it3-min5.w2v'
elif 'news' in VECTORS:
    if 'pos' in VECTORS:
        if 'sk' in VECTORS:
            EMB_PATH = ALL_EMB + 'news_pos_1_5.model'
        elif 'cbow'  in VECTORS:
            EMB_PATH = ALL_EMB + 'news_pos_0_5.model'
    elif 'ft' in VECTORS:
        if 'sk' in VECTORS:
            EMB_PATH = ALL_EMB + 'hypo_news/news_lemmas_ft_1_5.model'
        elif 'cbow'  in VECTORS:
            EMB_PATH = ALL_EMB + 'hypo_news/news_lemmas_ft_0_5.model'
else:
    print('Set up the path in configs.py')

RUWORDNET = 'input/resources/ruwordnet/'
if 'full' in VECTORS:
    SKIP_OOV = False
else:
    SKIP_OOV = True

if 'ft' in VECTORS:
    FT_EMB = EMB_PATH
else:
    try:
        FT_EMB = ALL_EMB + 'araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    except:
        FT_EMB = ALL_EMB + 'hypo_news/news_lemmas_ft_0_5.model'
        
# FT_EMB = ALL_EMB + '181_ruscorpora_fasttext/model.model'
OUT = 'output/'


