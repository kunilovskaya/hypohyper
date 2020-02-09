#!/bin/bash

python3.6 get_intrinsic_test.py
python3.6 format_data.py
python3.6 learn_projection.py
python3.6 get_hyper_vectors.py
python3.6 measure_sims.py

