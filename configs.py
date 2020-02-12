RANDOM_SEED = 0
# codename for emb: 'ft-araneum', w2v-rdt500, w2v-pos-ruscorpwiki, w2v-pos-araneum, ft-ruscorp, ft-araneum_full
# news-pos-sk, news-pos-sbow, ft-news-sk, ft-news-cbow, w2v-tayga-fpos5
VECTORS = 'w2v-tayga-fpos5' ## no full means that we filter out OOV at train time
POS = 'NOUN' # 'VERB'
MWE = True ## always TRUE wouldn't hurt
TEST = 'random' # static, provided, random, codalab

## strategies to improve performance
METHOD = 'base' # neg-hyp, neg-syn, deworded, corpus-informed25, base


if 'pos' in VECTORS:
    TAGS = True
else:
    TAGS = False
    print('Looks like you dont have tags in embeddings?')
    
if VECTORS == 'w2v-pos-araneum':
    EMB_PATH = '/home/u2/resources/emb/araneum_upos_skipgram_300_2_2018.bin'
elif VECTORS == 'w2v-pos-ruscorpwiki':
    EMB_PATH = '/home/u2/resources/emb/182_ruwikiruscorpora_upos_skipgram_300_2_2019/model.bin'
elif VECTORS == 'w2v-tayga-fpos5':
    EMB_PATH = '/home/u2/resources/emb/186_tayga-func_upos_skipgram_300_5_2019/model.bin'
elif 'ft-araneum' in VECTORS:
    EMB_PATH = '/home/u2/resources/emb/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
elif 'ft-ruscorp' in VECTORS:
    EMB_PATH = '/home/u2/resources/emb/181_ruscorpora_fasttext/model.model'
elif 'rdt' in VECTORS:
    EMB_PATH = '/home/lpvoid/masha/resources/emb/all.norm-sz500-w10-cb0-it3-min5.w2v'
elif 'news' in VECTORS:
    if 'pos' in VECTORS:
        if 'sk' in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_pos_1_5.model'
        elif 'cbow'  in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_pos_0_5.model'
    elif 'ft' in VECTORS:
        if 'sk' in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_lemmas_ft_1_5.model'
        elif 'cbow'  in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_lemmas_ft_0_5.model'
else:
    print('Set up the path in configs.py')

RUWORDNET = 'input/resources/ruwordnet/'
if 'full' in VECTORS:
    SKIP_OOV = False
else:
    SKIP_OOV = True
    
OOV_STRATEGY = 'top-hyper' ##'ft-vector', 'top-hyper'
MODE = 'single' # if you want to include vectors for main_words in MWE, replace single_wd with main;
# this this supposed to include vectors for main components of MWE only if this synset has no single_word representation or if MWE is found in vectors

if 'ft' in VECTORS:
    FT_EMB = EMB_PATH
else:
    try:
        FT_EMB = '/home/u2/resources/emb/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    except:
        FT_EMB = '/home/lpvoid/masha/resources/emb/hypo_news/news_lemmas_ft_0_5.model'
        
# FT_EMB = '/home/u2/resources/emb/181_ruscorpora_fasttext/model.model'
OUT = '/home/u2/proj/hypohyper/output/'

if 'ft' not in VECTORS and POS == 'VERB':
    TOPN = 1500
elif 'rdt' in VECTORS:
    TOPN = 1000
else:
    TOPN = 500
