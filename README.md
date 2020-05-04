KuKuPl team's contribution to the shared task at Dialogue Evaluation 2020:
[Taxonomy Enrichment for the Russian Language](https://competitions.codalab.org/competitions/22168)

The root folder has the code for the most successful approach -- a classifier trained on the customised word embeddings 
(which provide extended support for ruWordNet MWE) to predict synset ids.

The code for other, less successful approaches and their various settings, described in the paper(to be linked) and including 
- projection learning, 
- use of co-occurence statistics and 
- Hearst patterns

 has been moved to *trial_error* folder.

## Specially tailored embeddings we used
Download: http://vectors.nlpl.eu/repository/20/204.zip

Trained on lemmatised and UD-tagged:
- Araneum Russicum Maximum
- Russian Wikipedia
- Russian National Corpus (both the main and newspaper parts)
- Russian news corpus

In all the corpora, multi-word entities from ruWordNet were represented as one token. 
For example, *ящик_NOUN::из_ADP::картон_NOUN* and *давать_VERB::возможность_NOUN*