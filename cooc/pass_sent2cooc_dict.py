## to be used
## zcat "/media/u2/Seagate Expansion Drive/merged_ru/rus_araneum_maxicum.txt.gz" | python3 cooc/pass_sents.py | python3 cooc/get_cooc-stats.py
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser

from smart_open import open
from hyper_imports import preprocess_mwe
from configs import VECTORS, OUT, POS, TEST, METHOD, TAGS
import ahocorasick
import time
import json
from collections import defaultdict

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument('--ruthes_words', default='%sruWordNet_lemmas.txt' % OUT, help="68K words and phrases from ruWordNet")
    parser.add_argument('--ruthes_words', default='lists/tweaked_ruWordNet_%s_names_pos.txt' % POS,
                        help="68K words and phrases from ruWordNet ex научный_ADJ учреждение_NOUN")
    if TEST == 'codalab-pub':
        if POS == 'NOUN':
            parser.add_argument('--test', default='input/data/public_test/nouns_public.tsv', type=os.path.abspath)
        if POS == 'VERB':
            parser.add_argument('--test', default='input/data/public_test/verbs_public.tsv', type=os.path.abspath)
    if TEST == 'codalab-pr':
        if POS == 'NOUN':
            parser.add_argument('--test', default='input/data/private_test/nouns_private.tsv', type=os.path.abspath)
        if POS == 'VERB':
            parser.add_argument('--test', default='input/data/private_test/verbs_private.tsv', type=os.path.abspath)

    if TEST == 'provided':
        parser.add_argument('--test', default='lists/%s_%s_WORDS.txt' % (POS, TEST), type=os.path.abspath)
    # parser.add_argument('--words', default='output/mwe/ruWordNet_names_pos.txt', help="tagged 68K words and phrases from ruWordNet")
    args = parser.parse_args()
    
    start = time.time()
    
    ruthes_words = set()

    for line in open(args.ruthes_words):
        word = line.strip()
        # word = preprocess_mwe(word, tags=TAGS, pos=POS)
        word = ' ' + word  # avoid matching parts of words such as ель_NOUN, ад_NOUN, ток_NOUN, рота_NOUN, па_NOUN
        ruthes_words.add(word)

    print('%d ruthes lemmas read and tagged' % len(ruthes_words), file=sys.stderr)

    cooc_dict = defaultdict()

    for line in open(args.testwords):
        word = preprocess_mwe(line.strip(), tags=TAGS, pos=POS)
        word = ' ' + word  # avoid matching parts of words such as ель_NOUN, ад_NOUN, ток_NOUN, рота_NOUN, па_NOUN
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
    for line in sys.stdin:  # zcat corpus.txt.gz | python3 find_words.py
        res = line.strip()  # право::пациент_NOUN
        ## monitor progress
        count += 1
        if count % 10000000 == 0:
            print('%d lines passed to stats collecting, %.2f%% of the araneum corpus' % (count, count/748880899*100), file=sys.stderr) # 748880899

        for end_ind1, testword in auto1.iter(res):
            for end_ind2, ruthes_word in auto2.iter(res):
                if testword not in ruthes_word:  # avoid getting научный_ADJ учреждение_NOUN for hypo учреждение_NOUN
                    cooc_dict[testword][ruthes_word] += 1
                
    my_dict = defaultdict(list)
    number = 0
    for word in cooc_dict:  # number of test words; 1525 in private
        number += 1
        counter = 0
        ## it is already filtered thru ruWordNet, so no worries
        ## getting rid of the necessary annoying spaces
        predictions = [w.strip() for w in sorted(cooc_dict[word], key=cooc_dict[word].get, reverse=True)]
        word = word.strip()

        for w in predictions:
            if w == 'год_NOUN' or w == 'время_NOUN' or w == 'день_NOUN':  # too general concepts
                continue
            if word == w:
                continue
            if counter < 50:
                my_dict[word].append(w)
                counter += 1
            
            else:
                break
            # for each testword get a list of frequently cooccuring words, inluding 'детский_ADJ питание_NOUN'
        if number % 30 == 0:
            print(word, my_dict[word])
    first3pairs_ids = {k: my_dict[k] for k in list(my_dict)[:5]}
    print('Test2freq_cooc:', first3pairs_ids, file=sys.stderr)

    OUT_COOC = '%scooc/' % OUT
    os.makedirs(OUT_COOC, exist_ok=True)
    
    ## now I am getting 'детский_ADJ питание_NOUN' exclufing cases when test hypo is a subword
    outname = '%smerged5corp_%s_freq_cooc50_%s.json' % (OUT_COOC, TEST, POS)
    json.dump(my_dict, open(outname, 'w'))
    print('Written to: %s' % outname)

    print(len(my_dict.keys()))
    
    end = time.time()
    training_time = int(end - start)

    print('DONE: %s has run ===\nCo-occurence freqs_dict is written in %s minutes' %
          (os.path.basename(sys.argv[0]), str(round(training_time / 60))), file=sys.stderr)
    