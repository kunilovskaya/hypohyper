import argparse
import codecs
import json
from collections import defaultdict


# Consistent with Python 2

# python3 code/hypohyper/evaluate.py /home/u2/git/taxonomy-enrichment/data/sample_answers/nouns.tsv /home/u2/git/taxonomy-enrichment/data/sample_answers/nouns.tsv
## format АБСОРБЕНТ\t9980-N\tКАРТОГРАФ

def read_dataset(data_path, read_fn=lambda x: x, sep='\t'):
    vocab = defaultdict(list)
    with codecs.open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ## is this a .strip() alternative?
            line_split = line.replace("\n", '').split(sep)
            word = line_split[0]
            hypernyms = read_fn(line_split[1]) ## the optional lexicalizations of the synset_id in the 2nd col are ignored
            vocab[word].append(hypernyms) ## {АБСОРБЕНТ:[9980-N, 9981-N]}
    return vocab

### gets dicts {hypo:[hyper,hyper,hyper]}
def get_score(reference, predicted, k=10):
    ap_sum = 0
    rr_sum = 0

    for neologism in reference: ## loop over the keys
        ## for each hypo get (hypo, list-of-hyper) in ground_truth and predicted
        reference_hypernyms = reference.get(neologism, []) ## get [9980-N, 9981-N]; return empty list if the value does not exist
        predicted_hypernyms = predicted.get(neologism, [])
        # print(predicted_hypernyms)
        ap_sum += compute_ap(reference_hypernyms, predicted_hypernyms, k)
        # print([j for i in reference_hypernyms for j in i]) ## loop over the keys; is each key iterable?? hmmm
        ## ['1', '1', '3', '5', '4', '4', '-', 'N', '1', '3', '3', '5', '9', '9', '-', 'N', '1', '3', '3', '7', '5', '3', '-', 'N', '1', '1', '9', '6', '4', '6', '-', 'N', '1', '3', '1', '9', '1', '0', '-', 'N', '4', '1', '6', '7', '-', 'N', '1', '5', '0', '1', '4', '2', '-', 'N', '1', '0', '4', '8', '1', '3', '-', 'N', '1', '2', '1', '4', '2', '0', '-', 'N', '1', '2', '4', '1', '8', '1', '-', 'N']
        rr_sum += compute_rr(reference_hypernyms, predicted_hypernyms, k) ## [j for i in reference_hypernyms for j in i]
        
    return ap_sum / len(reference), rr_sum / len(reference) ## finding the mean over all hypos

## for this hypo: average the calculations for each predicted hyper; averaging favours the hits being at the top of the list
def compute_ap(actual, predicted, k=10): ## this gets golden and predicted lists of hypernyms for the current hypo
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    already_predicted = set()
    skipped = 0
    for i, p in enumerate(predicted):
        if p in already_predicted:
            skipped += 1
            continue
        for parents in actual:
            if p in parents:
                num_hits += 1.0
                score += num_hits / (i + 1.0 - skipped) ## how many out of seen-so-far are on the gold list; it is better to have hits at the top of the list
                already_predicted.update(parents)
                break
                
    return score / min(len(actual), k) ## this is where the averaging happens; did your system return less than 10 hypernyms? this function returns the smallest of two arguments


def compute_rr(true, predicted, k=10):
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0) ### I dont see how it is too different from MAP, except it doesn't not take into account duplicates on the list
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('predicted')
    args = parser.parse_args()

    reference = read_dataset(args.reference) # I don't have their json! lambda x: json.loads(x)
    submitted = read_dataset(args.predicted)
    if set(reference) != set(submitted):
        print(f"Not all words are presented in your file: {len(reference)} vs {len(submitted)}")
    mean_ap, mean_rr = get_score(reference, submitted, k=10)
    print("map: {0}\nmrr: {1}\n".format(mean_ap, mean_rr))


if __name__ == '__main__':
    main()
