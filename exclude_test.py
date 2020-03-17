# to get classifier's results on the provided test set in Table 1, exclude test words (lists/NOUN_provided_WORDS.txt)
# from the train (input/data/newtags_all_data_nouns.tsv)

# from smart_open import open

testwords = open('lists/NOUN_provided_WORDS.txt', 'r').readlines()
all_data = open('input/data/oldtags_all_data_nouns.tsv', 'r').readlines()

tagged = []
counter = 0
tot = 0
for w in testwords:
    w = w.strip().lower() + '_NOUN'
    tagged.append(w)
with open('input/data/notest_oldtags_all_data_nouns.tsv', 'w') as out:
    for line in all_data:
        hypo = line.split('\t')[0]
        if hypo in tagged:
            counter += 1
            continue
        else:
            tot += 1
            out.write(line)
print('Done! Train your classifier on all provided data for NOUN excluding %s test words' % len(testwords))
print('Checksum all (%d) - test word occuring in all (%d) = new_train (%d)' % (len(all_data), counter, tot))