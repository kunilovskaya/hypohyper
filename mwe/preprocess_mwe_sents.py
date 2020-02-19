## to be used
## zcat "/media/u2/Seagate Expansion Drive/merged_ru/araneum-rncwiki-news-rncP-pro.gz" | python3 mwe/preprocess_mwe_sents.py

import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)
import json
import time
from argparse import ArgumentParser
from collections import defaultdict
from collections import OrderedDict
from operator import itemgetter

from smart_open import open
from configs import OUT, POS


if __name__ == "__main__":
    OUT_MWE = '%smwe/' % OUT
    os.makedirs(OUT_MWE, exist_ok=True)
    
    parser = ArgumentParser()
    parser.add_argument('--tagged', default='%sruWordNet_names_pos.txt' % OUT_MWE, help="tagged 68K words and phrases from ruWordNet")
    args = parser.parse_args()

    start = time.time()
    
    words = []

    for line in open(args.tagged, 'r').readlines():
        line = line.strip()
        if line in ['ние_NOUN', 'к_NOUN', 'то_PRON', 'тот_DET', 'мочь_VERB', 'чать_VERB', 'нать_VERB', 'аз_NOUN', 'в_NOUN', 'дание_NOUN']:
            # print(line)
            continue
        else:
            words.append(line)
    words = set(words)

    print('%d lempos of words/phrases read' % len(words), file=sys.stderr)
    freq_dict = defaultdict(int)
    with open('%smwe_vectors_corpus_araneum-rncwiki-news-rncP-pro.gz' % OUT_MWE, 'a') as outfile:
        for line in sys.stdin: # zcat corpus.txt.gz | python3 this_script.py
            res = line.strip()
            for i in words:
                if i in res:
                    ## glue item
                    i_dup = i.replace(' ', '::')
                    res = res.replace(i, i_dup)
                    ## get a freq_dict: do I have enough to learn vectors for MWE?
                    freq_dict[i] += 1
                    ## white to file
                    outfile.write(res+'\n')
                    
    freq_dict_sort = OrderedDict(sorted(freq_dict.items(), key=itemgetter(1), reverse=True))
    first50pairs_ids = {k: freq_dict_sort[k] for k in list(freq_dict_sort)[:50]}
    print('Test2freq_cooc:', first50pairs_ids, file=sys.stderr)
    
    json.dump(freq_dict, open('%sfreq_araneum-rncwiki-news-rncP-pro_ruthes%ss.json' % (OUT_MWE, POS), 'w'))
    print('Written to: %sfreqs_araneum-rncwiki-news-rncP-pro_ruthes%ss.json' % (OUT_MWE, POS))
                    # print(line) ## this is fed into another script
                    # break
    end = time.time()
    training_time = int(end - start)

    print('DONE: %s has run ===\nSentences from the 5 corpora containing ruWordNet (glued) lempos written in %s minutes' %
          (os.path.basename(sys.argv[0]), str(round(training_time / 60))))

