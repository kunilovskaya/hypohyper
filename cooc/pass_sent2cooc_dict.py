## to be used
## zcat "/media/u2/Seagate Expansion Drive/merged_ru/rus_araneum_maxicum.txt.gz" | python3 cooc/pass_sents.py | python3 cooc/get_cooc-stats.py
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser

from smart_open import open
from hyper_imports import preprocess_mwe
from configs import VECTORS, OUT, POS, TEST, METHOD, TAGS, FILTER_1
import ahocorasick
import time
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ruthes_words', default='%sruWordNet_lemmas.txt' % OUT, help="68K words and phrases from ruWordNet")
    parser.add_argument('--testwords', default='%strains/%s_%s_%s_%s_WORDS.txt' % (OUT, VECTORS, POS, TEST, METHOD),
                        help="path to input word list")
    # parser.add_argument('--words', default='output/mwe/ruWordNet_names_pos.txt', help="tagged 68K words and phrases from ruWordNet")
    args = parser.parse_args()
    
    start = time.time()
    
    ruthes_words = set()

    for line in open(args.ruthes_words):
        word = line.strip()
        word = preprocess_mwe(word, tags=TAGS, pos=POS)
        ruthes_words.add(word)

    print('%d ruthes lemmas read and tagged' % len(ruthes_words), file=sys.stderr)

    cooc_dict = defaultdict()

    for line in open(args.testwords):
        word = preprocess_mwe(line.strip(), tags=TAGS, pos=POS)
        cooc_dict[word] = defaultdict(int)
        
        
    print('%d testwords read' % len(cooc_dict), file=sys.stderr)
    
    ## optimised iteration and string matching for getting relevant sentences
    auto1 = ahocorasick.Automaton()
    for substr in list(cooc_dict.keys()):  # listSubstrings
        auto1.add_word(substr, substr)
    auto1.make_automaton()
    
    auto2 = ahocorasick.Automaton()
    for substr in ruthes_words:  # listSubstrings
        auto2.add_word(substr, substr)
    auto2.make_automaton()
    
    count = 0
    for line in sys.stdin: # zcat corpus.txt.gz | python3 find_words.py
        res = line.strip() # право::пациент_NOUN
        ## monitor progress
        count += 1
        if count % 10000000 == 0:
            print('%d lines passed to stats collecting, %.2f%% of the araneum corpus' % (count, count/748880899*100), file=sys.stderr) # 748880899

        for end_ind1, testword in auto1.iter(res):
            for end_ind2, ruthes_word in auto2.iter(res):
                
                cooc_dict[testword][ruthes_word] += 1
                
                
    LIMIT = int(''.join([i for i in FILTER_1 if i.isdigit()]))
    my_dict = defaultdict(list)
    
    for word in cooc_dict:
        ## it is already filtered thru ruWordNet, so no worries
        predictions = [w for w in sorted(cooc_dict[word], key=cooc_dict[word].get, reverse=True)]
    
        counter = 0
    
        for w in predictions:
            if w == 'год_NOUN' or w == 'время_NOUN':  # too general concepts
                continue
            if word == w:
                continue
            if counter < LIMIT:
                my_dict[word].append(w)
                counter += 1
            else:
                break
            # for each testword get a list of frequently cooccuring words

    first3pairs_ids = {k: my_dict[k] for k in list(my_dict)[:2]}
    print('Test2freq_cooc:', first3pairs_ids, file=sys.stderr)

    OUT_COOC = '%scooc/' % OUT
    os.makedirs(OUT_COOC, exist_ok=True)

    json.dump(my_dict, open('%s%s_%s_freq_cooc%s_%s.json' % (OUT_COOC, VECTORS, TEST, LIMIT, POS), 'w'))
    print('Written to: %s%s_%s_freq_cooc%s_%s.json' % (OUT_COOC, VECTORS, TEST, LIMIT, POS))

    print(len(my_dict.keys()))
    
    end = time.time()
    training_time = int(end - start)

    print('DONE: %s has run ===\nCo-occurence freqs_dict is written in %s minutes' %
          (os.path.basename(sys.argv[0]), str(round(training_time / 60))), file=sys.stderr)
    