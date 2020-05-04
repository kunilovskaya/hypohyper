import sys
from smart_open import open
import argparse
from itertools import combinations 

def main():
    parser = argparse.ArgumentParser(description='Data conversion.')
    parser.add_argument('--input', help='Path to the input file.', required=True)
    parser.add_argument('--output', help='Path to the output file.', required=True)
    parser.add_argument('--tag', help='PoS tag to add')

    args = parser.parse_args()
    counter = 0
    with open(args.input, 'r')as f, open(args.output, 'w') as w:
        _ = next(f)
        for line in f:
            synset, hyponyms, parents, hypernyms = line.split('\t')
            hyponyms = hyponyms.strip().lower().split(',')
            hyponyms = [w.strip().replace(' ', '::') for w  in hyponyms]
            hypernyms = eval(hypernyms)
            hypernyms = [w.split(',') for w in hypernyms]
            hypernyms = [item for sublist in hypernyms for item in sublist]
            hypernyms = [w.strip().lower().replace(' ', '::') for w in hypernyms]
            if args.tag:
                hyponyms = [w + '_' + args.tag for w in hyponyms]
                hypernyms = [w + '_' + args.tag for w in hypernyms]
            for i in combinations(hyponyms, 2):
                hyp0 = i[0]
                hyp1 = i[1]
                w.write('\t'.join([str(counter), hyp0, hyp1, 'synonyms']) + '\n')
                counter += 1
            for hyponym in hyponyms:
                for hypernym in hypernyms:
                    w.write('\t'.join([str(counter), hyponym, hypernym, 'hypernyms']) + '\n')
                    counter += 1
                    # w.write('\t'.join([str(counter), hypernym, hyponym, 'hyponyms']) + '\n')
                    # counter += 1


if __name__ == '__main__':
    main()
