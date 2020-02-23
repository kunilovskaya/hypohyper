import sys
from collections import defaultdict
from smart_open import open
import json
from hyper_imports import preprocess_mwe
from configs import POS, TAGS
import ahocorasick

# corpus_file = open('/home/rgcl-dl/Projects/hypohyper/output/mwe/merged_mwe-glued_nofunct-punct_news-rncP5-pro.gz', 'r')
fixed_mwe = open('/home/rgcl-dl/Projects/hypohyper/output/mwe/merged_mwe-glued_nofunct-punct_fixed-mwe_news-rncP5-pro.gz', 'a')

# corpus_file = open('/home/u2/resources/corpora/head10000_news-taxonomy_temp.gz', 'r')
# fixed_mwe = open('/home/u2/TEMPmerged_mwe-glued_nofunct-punct_fixed-mwe_news-rncP5-pro.gz', 'a')

source = open('/home/rgcl-dl/Projects/hypohyper/output/ruWordNet_names.txt', 'r').readlines()
source_tagged = open('/home/rgcl-dl/Projects/hypohyper/output/mwe/ruWordNet_same-names_pos.txt', 'r').readlines()

## make a map to go from ЖРИЦА ЛЮБВИ to {'жрица::любви_NOUN' : 'жрица_NOUN::любви_NOUN'}
map = defaultdict()

for caps, tagged in zip(source, source_tagged):
    # print(caps, tagged)
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
for substr in map:  # iterate over dict keys
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
        print('%d lines processed, %.2f%% of the araneum only corpus' %
              (count1, count1 / 72704552 * 100), file=sys.stderr)
count = 0
for k, v in freq_dict.items():
    if v > 0:
        count += 1
        print(k)
    
json.dump(freq_dict, open('/home/u2/freq_fixed.json', 'w'))

print('Number of fixed MWE', count)

# corpus_file.close()
fixed_mwe.close()

