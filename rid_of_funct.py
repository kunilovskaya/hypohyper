from smart_open import open
import sys

def check_word(token, pos, nofunc=None, nopunct=True):
    outword = '_'.join([token, pos])
    if nofunc:
        if pos in nofunc:
            return None
    if nopunct:
        if pos == 'PUNCT':
            return None
    return outword


def num_replace(word):
    newtoken = 'x' * len(word)
    nw = newtoken + '_NUM'
    return nw

# corpus_file = open('/home/rgcl-dl/Projects/hypohyper/output/mwe/merged_mwe-glued_news-rncP5-pro.gz', 'r')
filtered = open('/home/rgcl-dl/Projects/hypohyper/output/mwe/merged_mwe-glued_nofunct-punct_news-rncP5-pro.gz', 'a')

functional = set('ADP AUX CCONJ DET PART PRON SCONJ PUNCT'.split())
SKIP_1_WORD = True
count = 0
for line in sys.stdin:
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
    count += 1
    if count % 1000000 == 0:
        print('%d lines processed, %.2f%% of the araneum only corpus' %
              (count, count / 72704552 * 100), file=sys.stderr)
    
    filtered.write(' '.join(good))
    filtered.write('\n')

# corpus_file.close()
filtered.close()