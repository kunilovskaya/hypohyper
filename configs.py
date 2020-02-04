RANDOM_SEED = 42

VECTORS = 'w2v_upos_araneum' ## arbitrary name of the embedding for formatting purposes: w2v_rdt500, w2v_upos_araneum, ft_ruscorp, ft_araneum_full
TAGS = True
POS = 'NOUN' # 'VERB'
MWE = True
EMB_PATH = '/home/u2/resources/emb/araneum_upos_skipgram_300_2_2018.vec.gz'
# EMB_PATH = '/home/u2/resources/emb/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
# EMB_PATH = '/home/u2/resources/emb/181_ruscorpora_fasttext/model.model'
# EMB_PATH = '/home/u2/resources/emb/all.norm-sz500-w10-cb0-it3-min5.w2v'
RUWORDNET = 'input/resources/ruwordnet/'
OOV_STRATEGY = 'top_hyper' ##'ft_vector', 'top_hyper', 'vec_on_fly', 'patterns'
FT_EMB = '/home/u2/resources/emb/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
# FT_EMB = '/home/u2/resources/emb/181_ruscorpora_fasttext/model.model'
OUT = '/home/u2/proj/hypohyper/output/'
MODE = 'main' # if you want to include vectors for main_words in MWE, replace single_wd with main;
# this this supposed to include vectors for main components of MWE only if this synset has no single_word representation or if MWE is found in vectors