import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(path):
    image = cv2.imread(path, 0)
    return image


class SIIMDataset(Dataset):

    def __init__(self,
                 csv_file,
                 root,
                 transform,
                 mode='train',
                 ):
        print(csv_file)
        df = pd.read_csv(csv_file, nrows=None)
        if mode != 'test':
            self.images = df['image'].values
        else:
            self.images = df['ImageId'].values
        self.root = root
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        if self.mode != 'test':
            image = os.path.join(self.root, 'train', image_name)
            mask = os.path.join(self.root, 'mask', image_name)

            image = load_image(image)
            mask = load_mask(mask)

            if self.transform:
                transform = self.transform(image=image, mask=mask)
                image = transform['image']
                mask = transform['mask']

            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)

            mask = mask / 255
            mask[mask >= 0.5] = 1.0
            mask[mask < 0.5] = 0.0
        else:
            image = os.path.join(self.root, 'test', image_name + '.png')
            # print(image)
            image = load_image(image)

            if self.transform:
                transform = self.transform(image=image)
                image = transform['image']

            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            mask = np.zeros((256, 256))

        return {
            "images": image,
            "targets": mask
        }
