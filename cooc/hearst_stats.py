#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser
from smart_open import open
from hyper_imports import preprocess_mwe
import json
import time
import re
from collections import defaultdict
from configs import VECTORS, OUT, TAGS, POS, TEST, METHOD
import ahocorasick

def hearst_hyper2(testword, sent, pat2, hyper_nr=2):
    hypos = set()
    m = re.search(pat2, sent)
    if m is not None:
        hyper = m.group(hyper_nr)
        hypos.add(m.group(3))

        try:
            hypos.add(m.group(4))
        except IndexError:
            pass
        try:
            hypos.add(m.group(5))
        except IndexError:
            pass
        try:
            hypos.add(m.group(6))
        except IndexError:
            pass
    
        if testword in hypos:
            return hyper
        else:
            return None


def hearst_hyper2a(testword, sent, pat2a, hyper_nr=2):
    hypos = set()
    m = re.search(pat2a, sent)
    
    if m is not None:
        hyper = m.group(hyper_nr)
        hypos.add(m.group(5))
        try:
            hypos.add(m.group(7))
        except IndexError:
            pass
        
        if testword in hypos:
            return hyper
        else:
            return None


def hearst_hyper4(testword, sent, pat4, hyper_nr=4):
    hypos = set()
    m = re.search(pat4, sent)
    if m is not None:
        hyper = m.group(hyper_nr)
        hypos.add(m.group(2))
        hypos.add(m.group(3))
        try:
            hypos.add(m.group(4))
        except IndexError:
            pass

        if testword in hypos:
            return hyper
        else:
            return None
        
def hearst_hyper5(testword, sent, pat5, hyper_nr=5):
    hypos = set()
    m = re.search(pat5, sent)
    if m is not None:
            hyper = m.group(hyper_nr)
            hypos.add(m.group(2))
            if m.group(3) == 'это_PRON':
                pass
            else:
                hypos.add(m.group(3))
                
            try:
                if m.group(3) not in ['сорт_NOUN', 'разновидность_NOUN', 'форма_NOUN', 'тип_NOUN','вид_NOUN','способ_NOUN']:
                    hypos.add(m.group(4))
            except IndexError:
                print(pat5)

            if testword in hypos:
                return hyper
            else:
                return None

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--testwords', default='%strains/%s_%s_%s_%s_WORDS.txt' % (OUT, VECTORS, POS, TEST, METHOD),
                        help="path to input word list")
    parser.add_argument('--ruthes_words', default='%smwe/ruWordNet_names_pos.txt' % OUT, help="path to words from WordNet")
    args = parser.parse_args()

    start = time.time()
    
    pat_dict = defaultdict()
    
    for line in open(args.testwords):
        tword = preprocess_mwe(line.strip(), tags=TAGS, pos=POS)
        pat_dict[tword] = []
    
    synset_words = set()

    for line in open(args.ruthes_words):
        line = line.strip()
        synset_words.add(line)

    print('%d testwords read' % len(pat_dict), file=sys.stderr)
    print('%d ruthes lemmas read' % len(synset_words), file=sys.stderr)
    
    # testword = [а-я]+_[A-Z]+(\:\:[а-я]+_[A-Z]+){0,4} ## if using new tagged corpus
    ## Such X as Y,[ Y,][ and/or Y]
    pat1_1 = r'(такой_DET\s([а-я]+_NOUN)\sкак_SCONJ\s([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat1_2 = r'(такой_DET\s([а-я]+_NOUN)\sкак_SCONJ\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat1_3 = r'(такой_DET\s([а-я]+_NOUN)\sкак_SCONJ\s([а-я]+_NOUN)\sили_CCONJ\s([а-я]+_NOUN))'
    pat1_4 = r'(такой_DET\s([а-я]+_NOUN)\sкак_SCONJ\s([а-я]+_NOUN)\s)'
    ## X, such as Y,[ Y,][ and/or Y]
    pat2_1 = r'(([а-я]+_NOUN)\s._PUNCT\sтакой_DET\sкак_SCONJ\s([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN),_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat2_2 = r'(([а-я]+_NOUN)\s._PUNCT\sтакой_DET\sкак_SCONJ\s([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat2_3 = r'(([а-я]+_NOUN)\s._PUNCT\sтакой_DET\sкак_SCONJ\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat2_4 = r'(([а-я]+_NOUN)\s._PUNCT\sтакой_DET\sкак_SCONJ\s([а-я]+_NOUN)\sили_CCONJ\s([а-я]+_NOUN))'
    pat2_5 = r'(([а-я]+_NOUN)\s._PUNCT\sтакой_DET\sкак_SCONJ\s([а-я]+_NOUN))'
    ## X: Y[, Y] and/or Y. (X: Y,[ Y,] and/or Y).
    pat3_1 = r'(([а-я]+_NOUN)\s:_PUNCT\s([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN),_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat3_2 = r'(([а-я]+_NOUN)\s:_PUNCT\s([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat3_3 = r'(([а-я]+_NOUN)\s:_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN))'
    pat3_4 = r'(([а-я]+_NOUN)\s:_PUNCT\s([а-я]+_NOUN)\sили_CCONJ\s([а-я]+_NOUN))'
    ## inverted order
    ## Y[, Y][(, а также)/(также как и)/и/или] прочий/другие/другим/других/о других X.
    ## Y[, Y] as well as other X.
    pat4_1 = r'(([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sдругой_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_2 = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\sтакже_ADV\sкак_SCONJ\sи_CCONJ\sдругой_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_3 = r'(([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sдругой_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_4 = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sдругой_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_5 = r'(([а-я]+_NOUN)\sили_CCONJ\s([а-я]+_NOUN)\sкак_SCONJ\sи_CCONJ\sдругой_ADJ{1}\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_6 = r'(([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sдругой_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    
    pat4_1a = r'(([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sпрочий_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_2a = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\sтакже_ADV\sкак_SCONJ\sи_CCONJ\sпрочий_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_3a = r'(([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sпрочий_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_4a = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sпрочий_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_5a = r'(([а-я]+_NOUN)\sили_CCONJ\s([а-я]+_NOUN)\sкак_SCONJ\sи_CCONJ\sпрочий_ADJ{1}\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat4_6a = r'(([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sпрочий_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    
    pat5_1 = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sдругой_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_2 = r'(([а-я]+_NOUN)\sили_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sдругой_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_3 = r'(([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sдругой_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_4 = r'(([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sдругой_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_5 = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sдругой_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'

    pat5_1a = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sпрочий_ADJ\s(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_2a = r'(([а-я]+_NOUN)\sили_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sпрочий_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_3a = r'(([а-я]+_NOUN)\sи_CCONJ\s([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\sпрочий_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_4a = r'(([а-я]+_NOUN)\s,_PUNCT\sа_CCONJ\sтакже_ADV\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sпрочий_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'
    pat5_5a = r'(([а-я]+_NOUN)\s,_PUNCT\s([а-я]+_NOUN)\s,_PUNCT\sкак_SCONJ\sи_CCONJ\sпрочий_ADJ\s([а-я]+_[A-Z]+\s){0,2}?(([а-я]+_ADJ\s)?[а-я]+_NOUN))'

    # Y — вид/тип/форма/разновидность/сорт X; Y (is a) kind/type/form/sort of X
    # самый_ADJ серьезный_ADJ проблема_NOUN -_PUNCT это_PRON человек_NOUN; паранойя_NOUN -_PUNCT это_PRON хороший_ADJ ._PUNCT
    pat6_1 = r'(([а-я]+_NOUN)\s\-_PUNCT\s(это_PRON\s)?(такой_ADJ\s)?(?:вид_NOUN|тип_NOUN|форма_NOUN|разновидность_NOUN|сорт_NOUN|способ_NOUN|стиль_NOUN)\s(([а-я]+_[A-Z]+\s)?[а-я]+_NOUN))'

    pat7_1 = r'((([а-я]+_ADJ\s)?[а-я]+_NOUN)\sвроде_ADP\s([а-я]+_NOUN)(\sи_CCONJ\s([а-я]+_NOUN))?)'

    pat_hyper2 = [pat1_1,pat1_2, pat1_3, pat2_1,pat2_2, pat2_3, pat2_4, pat2_5, pat3_1,pat3_2, pat3_3, pat3_4]
    pat_hyper2a = [pat7_1]
    pat_hyper4 = [pat4_1, pat4_2,pat4_3, pat4_4, pat4_5, pat4_6, pat4_1a, pat4_2a,pat4_3a, pat4_4a, pat4_5a, pat4_6a]
    pat_hyper5 = [pat5_1, pat5_2, pat5_3, pat5_4, pat5_5, pat6_1, pat5_1a, pat5_2a, pat5_3a, pat5_4a, pat5_5a]
    
    # optimised iteration and string matching
    auto = ahocorasick.Automaton()
    for substr in pat_dict:  # listSubstrings
        auto.add_word(substr, substr)
    auto.make_automaton()
    
   
    count = 0
    for line in sys.stdin:  # corpus, sys.stdin: zcat corpus.gz | python3 this_script.py
        sent = line.strip()
        for end_ind, word in auto.iter(sent):
            for pat in pat_hyper2:
                new_hyper = hearst_hyper2(word, sent, pat, hyper_nr=2)
                if new_hyper and new_hyper in synset_words:
                    pat_dict[word].append(new_hyper)
                    
            for pat in pat_hyper2a:
                new_hyper = hearst_hyper2(word, sent, pat, hyper_nr=2)
                if new_hyper and new_hyper in synset_words:
                    pat_dict[word].append(new_hyper)
                    
            for pat in pat_hyper4:
                new_hyper = hearst_hyper4(word, sent, pat, hyper_nr=4)
                if new_hyper and new_hyper in synset_words:
                    pat_dict[word].append(new_hyper)
                    
            for pat in pat_hyper5:
                new_hyper = hearst_hyper5(word, sent, pat, hyper_nr=5)
                if new_hyper and new_hyper in synset_words:
                    pat_dict[word].append(new_hyper)

        count += 1
        if count % 10000000 == 0:
            print('%d lines processed, %.2f%% of the araneum only corpus' %
                  (count, count / 748880899 * 100), file=sys.stderr)  # 748880899
                   
    for hypo, hypers in pat_dict.items():
        if len(pat_dict[hypo]) >= 1:
            print(hypo, hypers)

    OUT_COOC = '%scooc/' % OUT
    os.makedirs(OUT_COOC, exist_ok=True)

    print('We found Hearst hypers from ruWordNet for %d input words' % len([w for w in pat_dict if len(pat_dict[w]) >= 1]), file=sys.stderr)

    out = json.dump(pat_dict, open('%shearst-hypers_%s_%s_%s.json' % (OUT_COOC, VECTORS, POS, TEST), 'w'), ensure_ascii=False,
                    indent=4, sort_keys=True)

    end = time.time()
    training_time = int(end - start)

    print('DONE: %s has run ===\nHearst patterns detected hypernyms in %s seconds' %
          (os.path.basename(sys.argv[0]), str(round(training_time))), file=sys.stderr)