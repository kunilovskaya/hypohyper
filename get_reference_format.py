import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
from sklearn.model_selection import train_test_split
from math import isclose
from argparse import ArgumentParser


# read train file in format "SYNSET<TAB>SENSES<TAB>PARENTS<TAB>DEFINITION"
# return:
#     - :components: -- connected components of the graph of synsets
#                       (synsets are connected if they have senses expressed with the same word
#                        e.g. 'CLASS' as a room or a group of people (CLASS has two connected components))
#                       each connected component is simply a set of nodes
#     - :synset2word: -- a dictionary, keys are synsets, values are lists of words (senses) corresponding to a synset
#     - :word2parents: -- a dictionary, keys are words (senses), values are lists of parents of a corresponding synset
def get_data(train_file):
    word2parents = defaultdict(list)
    synset2parents = defaultdict(list)
    G = nx.Graph()
    word2synset = defaultdict(set)
    synset2word = defaultdict(set)
    for idx, line in enumerate(open(train_file)):
        if idx == 0:
            continue
        row = line.strip('\n').split('\t')
        words = row[1].split(', ')
        parents = row[2].strip("[]").split(', ')
        parents = sorted([r.strip("'") for r in parents])
        # why 126551-N	КРЕСТНЫЙ ОТЕЦ	['4544-N', '147272-N']	['ВЕРУЮЩИЙ В ХРИСТА, БРАТЬЯ ВО ХРИСТЕ, ХРИСТИАНИН,
        # and 126551-N	КРЕСТНЫЙ ОТЕЦ ['2553-N', '153536-N', '2453-N'] 'СОВЕРШЕННОЛЕТНИЕ ЛИЦА, СОВЕРШЕННОЛЕТНИЙ ГРАЖДАНИН, ВЗРОСЛЫЙ ЧЕЛОВЕК', 'МУЖИЧОНКА, МУЖЧИНА, МУЖИК,
        synset2parents[row[0]].append(parents)
        for w in words:
            word2parents[w].append(parents)
            word2synset[w].add(row[0])
            synset2word[row[0]].add(w)
            G.add_node(row[0])
            if len(word2synset[w]) > 1: # 107722-N	ДЛАНЬ, РУКА, РУЧКА, РУЧОНКА; 139821-N РУЧКА УПРАВЛЕНИЯ, РУКОЯТКА
                print(w)
                for n1, n2 in combinations(word2synset[w], 2):
                    G.add_edge(n1, n2)
    components = list(nx.connected_components(G)) # nx.connected_components(G) - is a generator (lazy iterator) over all connected components in G
    len_comp = [len(c) for c in components]
    return components, synset2word, word2parents, synset2parents


# get all senses belonging to a given component
def get_words(comp, synset2word):
    out = []
    for c in comp:
        out.extend(synset2word[c]) ## adding a list to a list without creating a nested structure
    return out


# split the data into train, dev, test
# returns words (senses) for each of subsets
# senses from connected synsets go to the same subset
def generate_split(components, synset2word, word2parents, partition=[0.8, 0.1, 0.1]):
    split_components = [[] for i in range(len(partition))]
    split_words = [[] for i in range(len(partition))]
    for i in range(len(components)):
        partition_interval = [(sum(partition[:i])*100, sum(partition[:i+1])*100) for i in range(len(partition))]
        bucket = np.argmax([p1 < (i % 100) <= p2 for (p1, p2) in partition_interval])
        words = get_words(components[i], synset2word)
        split_words[bucket].extend(words)

    split_words = [sorted(list(set(w))) for w in split_words]
    return split_words


# write data to file in format "SENSE<TAB>PARENT SYNSETS"
# the data is written in the format accepted by the evaluation script as reference
def write_data(words, word2parents, out_file):
    words = sorted(words)
    out = open(out_file, 'w')
    for w in words:
        for parents in word2parents[w]:
            out.write('%s\t%s\n' % (w, parents))
    out.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input', default='input/data/training_nouns.tsv', help='train data in format SYNSET<TAB>SENSES<TAB>PARENTS<TAB>DEFINITION')
    parser.add_argument('output', default='/home/u2/proj/hypohyper/train.tsv /home/u2/proj/hypohyper/dev.tsv /home/u2/proj/hypohyper/test.tsv', nargs='+', help='new file(s) to store the data')
    parser.add_argument('--split', default='0.8 0.1 0.1', nargs='+', help='size of train/dev/test splits. Has to sum to 1, e.g. "0.8 0.1 0.1"')
    args = parser.parse_args()
    
    if len(args.output) > 1:
        assert(args.split is not None), 'Multiple output files specified, but split not provided'
        assert(len(args.split) == len(args.output)), '{} subsets specified for split, {} output files'.format(len(args.split), len(args.output))
        partition = [float(i) for i in args.split]
        assert(isclose(sum(partition), 1)), 'Provided data splits do not sum to 1: {}'.format(str(args.split))
    
    components, synset2word, word2parents, _ = get_data(args.input)
    if args.split is not None:
        split_words = generate_split(components, synset2word, word2parents, partition=partition)
        for dataset, out_file in zip(split_words, args.output):
            write_data(dataset, word2parents, out_file)
    else:
        write_data(list(word2parents.keys()), word2parents, args.output[0])