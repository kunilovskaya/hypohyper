#! python3
# coding: utf-8
# USAGE: zcat "/media/u2/Seagate Expansion Drive/merged_ru/rus_araneum_maxicum.txt.gz" | python3 cooc/pass_sents.py | python3 cooc/get_cooc-stats.py
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser
from smart_open import open
from hyper_imports import preprocess_mwe
import json
import time
from configs import VECTORS, OUT, TAGS, POS, TEST, METHOD


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--testwords', default='%strains/%s_%s_%s_%s_WORDS.txt' % (OUT, VECTORS, POS, TEST, METHOD), help="path to input word list")
    parser.add_argument('--ruthes_words', default='%sruWordNet_lemmas.txt' % OUT, help="path to words from WordNet")
    args = parser.parse_args()

    start = time.time()
    
    words = {}

    for line in open(args.testwords):
        word = preprocess_mwe(line.strip(), tags=TAGS, pos=POS)
        words[word] = {}

    synset_words = set()

    for line in open(args.ruthes_words):
        word = preprocess_mwe(line.strip(), tags=TAGS, pos=POS)
        synset_words.add(line)
        
    print('%d testwords read' % len(words), file=sys.stderr)
    print('%d ruthes lemmas read' % len(synset_words), file=sys.stderr)

    for line in sys.stdin: ## corpus sentences??
        lemmas = set(line.strip().split())
        for word in words:
            if word in lemmas:
                for synset_word in synset_words:
                    if synset_word in lemmas:
                        if synset_word not in words[word]:
                            words[word][synset_word] = 0
                        words[word][synset_word] += 1
    
    OUT_COOC = '%scooc/' % OUT
    os.makedirs(OUT_COOC, exist_ok=True)
    
    print('We found data for %d input words' % len([w for w in words if len(words[w]) > 1]), file=sys.stderr)

    out = json.dump(words, open('%scooc-stats_%s_%s_%s.json' % (OUT_COOC, VECTORS, POS, TEST), 'w'), ensure_ascii=False, indent=4, sort_keys=True)
    # print(out)

    end = time.time()
    training_time = int(end - start)

    print('DONE: %s has run ===\nCo-occurence freqs_dict is written in %s minutes' %
          (os.path.basename(sys.argv[0]), str(round(training_time / 60))))

