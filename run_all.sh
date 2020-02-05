#!/bin/bash

python3.6 format_train.py
python3.6 learn_projection.py
# python3.6 test_projection.py
python3.6 get_hyper_vectors.py
python3.6 measure_sims.py

