#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
RUN_CONFIG=config.yml
RUN_CONFIG2=config2.yml


for fold in 0; do
    #stage 1
    LOGDIR=/raid/bac/kaggle/logs/siim/test/190805/unet34_256_multistages/fold_$fold/stage1/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --stages/data_params/train_csv=./csv/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/valid_$fold.csv:str \
        --verbose

    #stage 2
    LOGDIR2=/raid/bac/kaggle/logs/siim/test/190805/unet34_256_multistages/fold_$fold/stage2/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG2} \
        --logdir=$LOGDIR2 \
        --out_dir=$LOGDIR2:str \
        --stages/data_params/train_csv=./csv/train_$fold.csv:str \
        --stages/data_params/valid_csv=./csv/valid_$fold.csv:str \
        --model_params/pretrained=${LOGDIR}/checkpoints/best.pth:str \
        --verbose
done