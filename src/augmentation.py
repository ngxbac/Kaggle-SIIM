from albumentations import *

import itertools


def train_aug(image_size=224):
    return Compose([
        HorizontalFlip(),
        Normalize()
    ],p=1)


def valid_aug(image_size=224):
    return Compose([
        Normalize()
    ], p=1)


def test_tta(image_size):
    test_dict = {
        'normal': Compose([
            Resize(image_size, image_size)
        ]),
        # 'hflip': Compose([
        #     HorizontalFlip(p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
        # 'rot90': Compose([
        #     Rotate(limit=(90, 90), p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
    }

    return test_dict