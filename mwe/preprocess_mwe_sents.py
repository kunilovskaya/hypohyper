# to be used
# zcat "/media/u2/Seagate Expansion Drive/merged_ru/araneum-rncwiki-news-rncP-pro.gz"
# | python3 mwe/preprocess_mwe_sents.py
# zcat "/media/rgcl-dl/Seagate Expansion Drive/merged_ru/araneum-rncwiki-news-rncP-pro.gz"
# | python3 mwe/preprocess_mwe_sents.py
import os
import sys
import json
import time
from argparse import ArgumentParser
from collections import defaultdict
from collections import OrderedDict
from operator import itemgetter
from smart_open import open
import ahocorasick

if __name__ == "__main__":
    path1 = '../hypohyper/'
    path1 = os.path.abspath(str(path1))
    sys.path.append(path1)

    from configs import OUT, POS
    OUT_MWE = '%smwe/' % OUT
    os.makedirs(OUT_MWE, exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument('--tagged', default='%sruWordNet_names_pos.txt' % OUT_MWE,
                        help="tagged 68K words and phrases from ruWordNet")
    args = parser.parse_args()

    start = time.time()

    words = set()

    for line in open(args.tagged, 'r').readlines():
        line = line.strip()
        if line in ['ние_NOUN', 'к_NOUN', 'то_PRON', 'тот_DET', 'мочь_VERB', 'чать_VERB',
                    'нать_VERB', 'аз_NOUN', 'в_NOUN', 'дание_NOUN']:
            continue
        if len(line.split()) == 1:
            continue
        else:
            words.add(line)

    # optimised iteration and string matching
    auto = ahocorasick.Automaton()
    for substr in words:  # listSubstrings
        auto.add_word(substr, substr)
    auto.make_automaton()

    count = 0
    print('%d lempos of words/phrases read' % len(words), file=sys.stderr)
    freq_dict = defaultdict(int)
    with open('%smwe_vectors_corpus_araneum-rncwiki-news-rncP-pro.gz' % OUT_MWE, 'a') as outfile:
        # with gzip.open('/home/u2/temp/pro_lempos_ol.gz', 'rb') as f:
        for line in sys.stdin:  # f: # zcat corpus.txt.gz | python3 this_script.py
            res = line.strip()
            # res = line.decode("utf-8").strip()
            count += 1
            if count % 10000000 == 0:
                print('%d lines processed, %.2f%% of the araneum only corpus' %
                      (count, count / 748880899 * 100), file=sys.stderr)  #
            seen = set()

            for end_ind, found in auto.iter(res):
                # print(found)
                if found not in seen:
                    seen.add(found)

                    found_dup = found.replace(' ', '::')
                    res = res.replace(found, found_dup)
                    # get a freq_dict: do I have enough to learn vectors for MWE?
                    freq_dict[found] += 1
                    # white to file
            outfile.write(res + '\n')
            
    freq_dict_sort = OrderedDict(sorted(freq_dict.items(), key=itemgetter(1), reverse=True))
    first50pairs_ids = {k: freq_dict_sort[k] for k in list(freq_dict_sort)[:50]}
    print('Test2freq_cooc:', first50pairs_ids, file=sys.stderr)

    json.dump(freq_dict, open('%sfreq_araneum-rncwiki-news-rncP-pro_ruthes%ss.json'
                              % (OUT_MWE, POS), 'w'))
    print('Written to: %sfreqs_araneum-rncwiki-news-rncP-pro_ruthes%ss.json'
          % (OUT_MWE, POS), file=sys.stderr)

    end = time.time()
    training_time = int(end - start)

    print('DONE: %s has run ===\nSentences from the 5 corpora containing '
          'ruWordNet (glued) lempos written in %s minutes' %
          (os.path.basename(sys.argv[0]), str(round(training_time / 60))), file=sys.stderr)
