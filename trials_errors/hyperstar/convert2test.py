import sys
from smart_open import open
import argparse

def main():
    parser = argparse.ArgumentParser(description='Data conversion.')
    parser.add_argument('--input', help='Path to the input file.', required=True)
    parser.add_argument('--output', help='Path to the output file.', required=True)
    parser.add_argument('--tag', help='PoS tag to add')

    args = parser.parse_args()
    with open(args.input, 'r')as f, open(args.output, 'w') as w:
        for line in f:
            word = line.strip().lower()
            if args.tag:
                word = word + '_' + args.tag
            w.write('\t'.join([word, 'None']) + '\n')

if __name__ == '__main__':
    main()
