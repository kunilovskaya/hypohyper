#! python3
# coding: utf-8

from argparse import ArgumentParser
import sys
from smart_open import open
import json

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--datafile', required=True, help="JSON with co-occurrence data")
    args = parser.parse_args()

    with open(args.datafile, 'rb') as f:
        data = json.loads(f.read())

    for word in data:
        predictions = [w for w in sorted(data[word], key=data[word].get, reverse=True)[:10]]
        hypernyms = json.dumps(predictions, ensure_ascii=False)
        print('\t'.join([word, hypernyms]))
