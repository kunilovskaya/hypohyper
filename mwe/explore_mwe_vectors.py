import argparse
import sys
from smart_open import open

from trials_errors.configs import POS, EMB_PATH
from trials_errors.hyper_imports import load_embeddings

parser = argparse.ArgumentParser()
parser.add_argument('--tagged', default='lists/ruWordNet_%s_names_pos.txt' % POS,
                    help="tagged 68K words and phrases from ruWordNet")
args = parser.parse_args()

words = set()

for line in open(args.tagged, 'r').readlines():
    line = line.strip()
    if line in ['ние_NOUN', 'к_NOUN', 'то_PRON', 'тот_DET', 'мочь_VERB', 'чать_VERB',
                'нать_VERB', 'аз_NOUN', 'в_NOUN', 'дание_NOUN']:
        continue
    if len(line.split()) == 1:
        continue
    else:
        line = line.replace(' ','::')
        words.add(line)

print('Current embedding model:', EMB_PATH.split('/')[-1], file=sys.stderr)
emb = load_embeddings(EMB_PATH)

counter = 0
for word in words:
    if word in emb.vocab:
        # print('часть_NOUN::тело_NOUN' in emb.vocab)
        counter += 1
        
print('Ratio of MWE in dedicated emb to all MWE in ruWordNet: %.2f%%' % (counter/len(words)*100))
print('Absolute nums: %d, %d' % (counter,len(words)))
