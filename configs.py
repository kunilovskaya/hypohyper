RANDOM_SEED = 42
# codename for embeddings: 'ft_araneum_skip', w2v_rdt500, w2v_upos_araneum, ft_ruscorp_skip, ft_araneum_full, ft_news_0, w2v_upos_news_0
# news_pos_sk, news_pos_sbow, ft_news_sk, ft_news_cbow
VECTORS = 'w2v_upos_araneum'
POS = 'VERB' # 'VERB'
MWE = True

if 'pos' in VECTORS:
    TAGS = True
elif 'ft' in VECTORS:
    TAGS = False
else:
    print('Do you have tags in the embeddings?')
    
if VECTORS == 'w2v_upos_araneum':
    EMB_PATH = '/home/u2/resources/emb/araneum_upos_skipgram_300_2_2018.bin'
if 'ft_araneum' in VECTORS:
    EMB_PATH = '/home/u2/resources/emb/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
if 'news' in VECTORS:
    if 'pos' in VECTORS:
        if 'sk' in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_pos_1_5.model'
        if 'cbow'  in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_pos_0_5.model'
    if 'ft' in VECTORS:
        if 'sk' in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_lemmas_ft_1_5.model'
        if 'cbow'  in VECTORS:
            EMB_PATH = '/home/lpvoid/masha/resources/emb/hypo_news/news_lemmas_ft_0_5.model'
else:
    print('Set up the path in configs.py')
    
# EMB_PATH = '/home/u2/resources/emb/181_ruscorpora_fasttext/model.model'
# EMB_PATH = '/home/u2/resources/emb/all.norm-sz500-w10-cb0-it3-min5.w2v'
RUWORDNET = 'input/resources/ruwordnet/'
SKIP_OOV = True
OOV_STRATEGY = 'ft_vector' ##'ft_vector', 'top_hyper', 'vec_on_fly', 'patterns'
FT_EMB = '/home/u2/resources/emb/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
# FT_EMB = '/home/u2/resources/emb/181_ruscorpora_fasttext/model.model'
OUT = '/home/u2/proj/hypohyper/output/'
MODE = 'single' # if you want to include vectors for main_words in MWE, replace single_wd with main;
# this this supposed to include vectors for main components of MWE only if this synset has no single_word representation or if MWE is found in vectors