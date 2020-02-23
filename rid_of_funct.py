from smart_open import open

def check_word(token, pos, nofunc=None, nopunct=True, noshort=True, stopwords=None):
    outword = '_'.join([token, pos])
    if nofunc:
        if pos in nofunc:
            return None
    if nopunct:
        if pos == 'PUNCT':
            return None
    if stopwords:
        if token in stopwords:
            return None
    if noshort:
        if len(token) < 2:
            return None
    return outword


def num_replace(word):
    newtoken = 'x' * len(word)
    nw = newtoken + '_NUM'
    return nw

corpus_file = open('/home/rgcl-dl/Projects/hypohyper/output/mwe/merged_mwe-glued_news-rncP5-pro.gz', 'r')
filtered = open('/home/rgcl-dl/Projects/hypohyper/output/mwe/merged_mweglued_nofunct-punct_news-rncP5-pro.gz', 'a')

functional = set('ADP AUX CCONJ DET PART PRON SCONJ PUNCT'.split())
SKIP_1_WORD = True
for line in corpus_file:
    res = line.strip().split()
    good = []
    for w in res:
        if '::' in w:
            checked_word = w
        else:
            (token, pos) = w.split('_')
            checked_word = check_word(token, pos, nofunc=functional)   # Can feed stopwords list
            if not checked_word:
                continue
            if pos == 'NUM' and token.isdigit():  # Replacing numbers with xxxxx of the same length
                checked_word = num_replace(checked_word)
        good.append(checked_word)
    if SKIP_1_WORD:  # May be, you want to filter out one-word sentences
        if len(good) < 2:
            continue
    filtered.write(' '.join(good))
    filtered.write('\n')

corpus_file.close()
filtered.close()