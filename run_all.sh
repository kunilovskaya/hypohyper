#!/bin/bash

python3 code/hypohyper/format_train.py
python3 code/hypohyper/learn_projection.py
python3 code/hypohyper/test_projection.py
python3 code/hypohyper/get_hyper_vectors.py


