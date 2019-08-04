import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
import glob
import click
from tqdm import *
import cv2

from models import *
from augmentation import *
from dataset import SIIMDataset
from utils import *


device = torch.device('cuda')


def predict(model, loader):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = Ftorch.sigmoid(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)
            mask = dct['targets'].numpy()
            gts.append(mask)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts


def dice(outputs, targets, eps: float = 1e-7):
    intersection = np.sum(targets * outputs)
    sum_ = np.sum(targets) + np.sum(outputs) + eps

    return (2 * intersection + eps) / sum_


def threshold_search(preds, gts):
    scores = []
    ths = np.arange(0.01, 1, 0.01)
    for th in tqdm(ths, total=len(ths)):
        pred_bin = (preds > th).astype(np.float32)
        dice_scores = dice(pred_bin, gts)
        scores.append(dice_scores)

    best_score = np.max(scores)
    best_th = ths[np.argmax(scores)]
    return best_score, best_th


def predict_valid():
    test_csv = './csv/valid_0.csv'

    log_dir = f"/raid/bac/kaggle/logs/siim/test/190730/unet34/fold_0/"
    root = "/raid/data/kaggle/siim/siim256/"

    ckp = os.path.join(log_dir, "checkpoints/best.pth")
    model = Unet(
        encoder_name="resnet34",
        activation='sigmoid',
        classes=1
    )

    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("*" * 50)
    print(f"checkpoint: {ckp}")
    # Dataset
    dataset = SIIMDataset(
        csv_file=test_csv,
        root=root,
        transform=valid_aug()
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
    )

    preds, gts = predict(model, loader)

    threshold = 0.900000
    min_size = 3500

    loss, dice, dice_neg, dice_pos = \
        compute_metric(gts, preds)

    print('\n')
    print('loss     = %0.5f\n' % (loss))
    print('dice_neg = %0.5f\n' % (dice_neg))
    print('dice_pos = %0.5f\n' % (dice_pos))
    print('dice     = %0.5f\n' % (dice))
    print('\n')

    kaggle_score, kaggle_neg_score, kaggle_pos_score = \
        compute_kaggle_lb(gts, preds, threshold, min_size)

    print('kaggle_neg_score = %0.5f\n' % (kaggle_neg_score))
    print('kaggle_pos_score = %0.5f\n' % (kaggle_pos_score))
    print('kaggle_score     = %0.5f\n' % (kaggle_score))

    best_score, best_th = threshold_search(preds, gts)
    print(f"Best score {best_score}, best_threshold: {best_th}")

    os.makedirs("./prediction/unet34/fold_0/", exist_ok=True)
    np.save(f"./prediction/unet34/fold_0/valid.npy", preds)


def predict_test():
    test_csv = './csv/test.csv'

    log_dir = f"/raid/bac/kaggle/logs/siim/test/190730/unet34/fold_0/"
    root = "/raid/data/kaggle/siim/siim256/"

    ckp = os.path.join(log_dir, "checkpoints/best.pth")
    model = Unet(
        encoder_name="resnet34",
        activation='sigmoid',
        classes=1
    )

    checkpoint = torch.load(ckp)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("*" * 50)
    print(f"checkpoint: {ckp}")
    # Dataset
    dataset = SIIMDataset(
        csv_file=test_csv,
        root=root,
        transform=valid_aug(),
        mode='test'
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
    )

    preds, gts = predict(model, loader)

    threshold = 0.21
    min_size = 3500

    encoded_pixels = []
    for pred in preds:
        # import pdb
        # pdb.set_trace()
        pred = pred[0]
        if pred.shape != (1024, 1024):
            pred = cv2.resize(pred, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        pred, num_predict = post_process(pred, threshold, min_size)

        if num_predict == 0:
            encoded_pixels.append('-1')
        else:
            r = run_length_encode(pred)
            encoded_pixels.append(r)

    df = pd.read_csv(test_csv)
    df['EncodedPixels'] = encoded_pixels
    os.makedirs("./prediction/unet34/fold_0/", exist_ok=True)
    df.to_csv(f"./prediction/unet34/fold_0/submission_th{threshold}_more_augs.csv", index=False)


if __name__ == '__main__':
    predict_test()
    # predict_valid()
