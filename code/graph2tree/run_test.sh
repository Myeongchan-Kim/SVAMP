#!/usr/bin/env bash

python -m src.test_main \
    -mode test \
    -run_name run_cv_mawps \
    -dataset cv_mawps \
    -gpu -1 \
    -outputs \
    -results \
    -embedding random \
    -no-full_cv
