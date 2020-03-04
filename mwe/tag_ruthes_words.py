import os, sys

path1 = '../hypohyper/'
path1 = os.path.abspath(str(path1))
sys.path.append(path1)

from argparse import ArgumentParser
from ufal.udpipe import Model, Pipeline
from smart_open import open
from configs import OUT, POS

def process(pipeline, text=None):
    
    # обрабатываем текст, получаем результат в формате conllu:
    processed = pipeline.process(text)

    # пропускаем строки со служебной информацией:
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    tagged0 = [w.split('\t') for w in content if w]
    res = []
    for t in tagged0:
        if len(t) != 10:
            print(t)
            continue
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        res.append('%s_%s' % (lemma, pos))

    return res

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--words', default='lists/ruWordNet_%s_names.txt' % POS, help="ruWORDNET words")
    args = parser.parse_args()

    words = set()

    udpipe_filename = '/home/u2/tools/udpipe/udpipe_syntagrus.model'
    model = Model.load(udpipe_filename)
    
    process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    
    lines = open(args.words, 'r').readlines()
    
    ## temporary setting: I need the tagged list of the same length!
    # count = set()
    count = []
    with open('lists/ruWordNet_%s_same-names_pos.txt' % POS, 'w') as out:
    
        for id, line in enumerate(lines):
            word = line.strip().lower()
            # print(word.strip())

            output = process(process_pipeline, text=word)
            string = ' '.join(output) + '\n'
            if string in count:
                print(word)
            count.append(string)
            out.write(string)
            if id != 0 and id % 1000 == 0:
                print('%s words and phrases processed' % id)

    print('Items tagged %d' % len(count))

    
    