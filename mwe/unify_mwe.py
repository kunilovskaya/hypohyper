import sys
from collections import defaultdict
from smart_open import open
import json
from trials_errors.hyper_imports import preprocess_mwe
from trials_errors.configs import POS, TAGS
import ahocorasick

fixed_mwe = open('output/mwe/merged_mwe-glued_nofunct-punct_fixed-mwe_news-rncP5-pro.gz', 'a')

source = open('lists/ruWordNet_%s_names.txt' % POS, 'r').readlines()
source_tagged = open('lists/ruWordNet_%s_same-names_pos.txt' % POS, 'r').readlines()

## make a map to go from ЖРИЦА ЛЮБВИ to {'жрица::любви_NOUN' : 'жрица_NOUN::любовь_NOUN'}
map = defaultdict()

for caps, tagged in zip(source, source_tagged):
    if ' ' in caps:
        old = preprocess_mwe(caps, tags=TAGS, pos=POS)
        new = tagged.replace(' ', '::')
        map[old] = new
   
first_pairs = {k: map[k] for k in list(map)[:10]}
print('First few matched items:', first_pairs, file=sys.stderr)

print('Checksums:', len(source), len(source_tagged))
print('Num of MWE', len(map))

# optimised iteration and string matching
auto = ahocorasick.Automaton()
for substr in map:  # iterate over dict keys = жрица::любви_NOUN
    # print(substr)
    auto.add_word(substr, substr)
auto.make_automaton()

freq_dict = defaultdict(int)
count1 = 0
for line in sys.stdin:
    res = line.strip() #.split()

    for end_ind, found in auto.iter(res):
        res = res.replace(found, map[found])
        # get a freq_dict: do I have enough to learn vectors for MWE?
        freq_dict[found] += 1
        
    fixed_mwe.write(res + '\n')
    count1 += 1
    if count1 % 1000000 == 0:
        print('%d lines processed, %.2f%% of the merged corpus' %
              (count1, count1 / 72704552 * 100), file=sys.stderr)
count = 0
for k, v in freq_dict.items():
    if v > 0:
        count += 1
        print(k)

print('Number of fixed MWE', count)

json.dump(freq_dict, open('/home/rgcl-dl/Projects/hypohyper/freq_fixed.json', 'w'))

fixed_mwe.close()

