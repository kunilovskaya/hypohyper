import json
# from smart_open import open
from collections import defaultdict
data = json.load(open('/home/u2/git/hypohyper/output/cooc/hearst-hypers_merged-news-taxonomy-ruscorpwiki-rncP-pro_NOUN_provided.json', 'r'))
counter = 0
mwe = 0
all = 0
# for pair in data:
#     print(pair)
new_dict = defaultdict(list)
for k,v in data.items():
    # if k in ['абрикос_NOUN', 'шашка_NOUN', 'шаттл_NOUN', 'шатун_NOUN', 'частушка_NOUN']:
    # print(k, v)
    all += 1
    k_new = k.strip()
    if len(v) != 0:
        counter += 1
    for i in v:
        i = i.strip()
        # print(k,v)
    for i in set(v):
        if ' ' in i:
            i = '::'.join(i.split())
            print(i)
            mwe += 1
            new_dict[k_new].append(i)
# for k,v in new_dict.items():
#     print(k,v)
#
# print(len(new_dict))
#
json.dump(new_dict, open('/home/u2/git/hypohyper/output/cooc/hearst-hypers_merged-news-taxonomy-ruscorpwiki-rncP-pro_NOUN_provided_mwe.json', 'w'))
# print('%d MWE found' % len(data))
# print('Freq > 10: %d synset words' % counter)

print('found Hearst hyponyms for %d, inc. %d MWE out of %d' % (counter, mwe, all))