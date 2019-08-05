#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_CONFIG=config.yml


for fold in 0; do
    LOGDIR=/raid/bac/kaggle/logs/siim/test/190805/unet34_1024/fold_$fold/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --stages/data_params/train_csv=./csv/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/valid_$fold.csv:str \
        --verbose
done