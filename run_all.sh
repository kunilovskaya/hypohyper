#!/bin/bash

python3 format_train.py
python3 learn_projection.py
python3 test_projection.py
python3 get_hyper_vectors.py
python3 vectorise_ruwordnet.py --mode single_wd
python3 measure_sims.py

