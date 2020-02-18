#! python3
# coding: utf-8
# USAGE: zcat /home/u2/temp/pro_lempos_ol.gz | python3 mwe/mwe_generate_corpus_stats.py | python3 get_corp_info/predict_cooccurrences.py > /home/u2/temp/out.json
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser
from smart_open import open
from hyper_imports import preprocess_mwe
import json
from configs import VECTORS, OUT, TAGS, POS, TEST, METHOD


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--words', default='%strains/%s_%s_%s_%s_WORDS.txt' % (OUT, VECTORS, POS, TEST, METHOD), help="path to input word list")
    parser.add_argument('--synsets', default='%sruWordNet_lemmas.txt' % OUT, help="path to words from WordNet")
    args = parser.parse_args()

    words = {}

    for line in open(args.words):
        word = preprocess_mwe(line.strip(), tags=TAGS, pos=POS)
        words[word] = {}

    synset_words = set()

    for line in open(args.synsets):
        word = preprocess_mwe(line.strip(), tags=TAGS, pos=POS)
        synset_words.add(word)
    print('%d words read' % len(words), file=sys.stderr)
    print('%d synset words read' % len(synset_words), file=sys.stderr)

    for line in sys.stdin: ## corpus sentences??
        lemmas = set(line.strip().split())
        for word in words:
            if word in lemmas:
                for synset_word in synset_words:
                    if synset_word in lemmas:
                        if synset_word not in words[word]:
                            words[word][synset_word] = 0
                        words[word][synset_word] += 1

    print('We found data for %d input words' % len([w for w in words if len(words[w]) > 1]), file=sys.stderr)
    out = json.dumps(words, ensure_ascii=False, indent=4, sort_keys=True) ## this is the JSON I have?
    print(out)
