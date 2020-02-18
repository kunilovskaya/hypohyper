#!/bin/bash

python3.6 format_data.py
python3.6 learn_projection.py
python3.6 get_hyper_vectors.py
# if you want to use corp-info FILTER_1 generate the dictionaries with frequently cooccuring words
# it is only available for news-pos-cbow, w2v-pos-ruscorpwiki as VECTORS
python3.6 predict_synsets.py
python3.6 intrinsic_evaluate.py

