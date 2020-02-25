import json
from smart_open import open
from collections import defaultdict
data = json.load(open('/home/u2/git/hypohyper/output/cooc/w2v-pos-araneum_provided_freq_cooc25_NOUN.json', 'r'))
counter = 0
# for pair in data:
#     print(pair)
new_dict = defaultdict(list)
for k,v in data.items():
    # if k in ['абрикос_NOUN', 'шашка_NOUN', 'шаттл_NOUN', 'шатун_NOUN', 'частушка_NOUN']:
    # print(k)
    k_new = k.strip()
    for i in v:
        i = i.strip()
        # print(k,v)
#     for i in set(v):
        new_dict[k_new].append(i)
# for k,v in new_dict.items():
#     print(k,v)
#
# print(len(new_dict))
#
json.dump(new_dict, open('/home/u2/git/hypohyper/output/cooc/w2v-pos-araneum_provided_freq_cooc25_NOUN0.json', 'w'))
# print('%d MWE found' % len(data))
# print('Freq > 10: %d synset words' % counter)