RANDOM_SEED = 42
# codename for embeddings: 'ft_araneum', w2v_rdt500, w2v_pos_araneum, ft_ruscorp, ft_araneum_full
# news_pos_sk, news_pos_sbow, ft_news_sk, ft_news_cbow
VECTORS = 'w2v_pos_araneum' ## no full means that we filter out OOV at train time
POS = 'NOUN' # 'VERB'
MWE = True

if 'pos' in VECTORS:
    TAGS = True
else:
    TAGS = False
    print('Looks like you dont have tags in embeddings?')
    
if VECTORS == 'w2v_pos_araneum':
    EMB_PATH = '/home/u2/resources/emb/araneum_upos_skipgram_300_2_2018.bin'
elif 'ft_araneum' in VECTORS:
    EMB_PATH = '/home/u2/resources/emb/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
elif 'ft_ruscorp' in VECTORS:
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
    
OOV_STRATEGY = 'ft_vector' ##'ft_vector', 'top_hyper', 'vec_on_fly', 'patterns'
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

