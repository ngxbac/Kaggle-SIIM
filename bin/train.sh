#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
RUN_CONFIG=config.yml


for fold in 0; do
    LOGDIR=/raid/bac/kaggle/logs/siim/test/190730/unet34/fold_$fold/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --stages/data_params/train_csv=./csv/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/valid_$fold.csv:str \
        --verbose
done