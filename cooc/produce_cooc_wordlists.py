import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)


from argparse import ArgumentParser
from smart_open import open
import json
from collections import defaultdict

from configs import VECTORS, OUT, POS, TEST, FILTER_1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--datafile', default='%scooc/cooc-stats_%s_%s_%s.json' % (OUT, VECTORS, POS, TEST), help="JSON with co-occurrence data")
    args = parser.parse_args()

    with open(args.datafile, 'rb') as f:
        data = json.loads(f.read())
    LIMIT = int(''.join([i for i in FILTER_1 if i.isdigit()]))
    my_dict = defaultdict(list)
    for word in data:
        ## it is already filtered thru ruWordNet, so no worries
        predictions = [w for w in sorted(data[word], key=data[word].get, reverse=True)]

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