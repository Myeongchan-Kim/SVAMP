#!/usr/bin/env bash

python -m src.main \
    -mode train \
    -gpu 0 \
    -embedding roberta \
    -emb_name roberta-base \
    -encoder_layers 2 \
    -decoder_layers 2 \
    -d_model 64 \
    -d_ff 256 \
    -batch_size 4 \
    -epochs 2 \
    -dataset cv_mawps \
    -full_cv \
    -save_model \
    -run_name run_cv_mawps