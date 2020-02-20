## to be used
## zcat "/media/u2/Seagate Expansion Drive/merged_ru/rus_araneum_maxicum.txt.gz" | python3 cooc/pass_sents.py | python3 cooc/get_cooc-stats.py
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser

from smart_open import open
from hyper_imports import preprocess_mwe
from configs import VECTORS, RUWORDNET, OUT, POS, TEST, METHOD


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--words', default='%sruWordNet_lemmas.txt' % OUT, help="68K words and phrases from ruWordNet")
    # parser.add_argument('--words', default='output/mwe/ruWordNet_names_pos.txt', help="tagged 68K words and phrases from ruWordNet")
    args = parser.parse_args()

    words = set()

    for line in open(args.words):
        word = line.strip()
        word = preprocess_mwe(word, tags=True, pos=POS)
        words.add(word)

    print('%d ruthes lemmas read' % len(words), file=sys.stderr)
    count = 0
    for line in sys.stdin: # zcat corpus.txt.gz | python3 find_words.py
        res = set(line.strip().split()) # право::пациент_NOUN
        ## monitor progress
        count += 1
        if count % 10000000 == 0:
            print('%d lines passed to stats collecting, %.2f%% of the araneum only corpus' % (count, count/748880899*100))
            
        for w in words:
            if w in res:
                print(line) ## this is fed into another script
                break
