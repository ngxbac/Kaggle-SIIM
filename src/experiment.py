from collections import OrderedDict
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
import random
from catalyst.dl.experiment import ConfigExperiment
from dataset import *
from augmentation import train_aug, valid_aug


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage == "stage0":
            if hasattr(model_, 'freeze'):
                model_.freeze()
                print("Freeze backbone model !!!")
            else:
                for param in model_.encoder.parameters():
                    param.requires_grad = False
                print("Freeze backbone model !!!")

        else:
            if hasattr(model_, 'unfreeze'):
                model_.unfreeze()
                print("Unfreeze backbone model !!!")
            else:
                for param in model_.encoder.parameters():
                    param.requires_grad = True
                print("Unfreeze backbone model !!!")

        return model_

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        """
        image_key: 'id'
        label_key: 'attribute_ids'
        """

        image_size = kwargs.get("image_size", 320)
        train_csv = kwargs.get('train_csv', None)
        valid_csv = kwargs.get('valid_csv', None)
        root = kwargs.get('root', None)

        if train_csv:
            transform = train_aug(image_size)
            train_set = SIIMDataset(
                csv_file=train_csv,
                root=root,
                transform=transform,
                mode='train',
            )
            datasets["train"] = train_set

        if valid_csv:
            transform = valid_aug(image_size)
            valid_set = SIIMDataset(
                csv_file=valid_csv,
                root=root,
                transform=transform,
                mode='train',
            )
            datasets["valid"] = valid_set

        return datasets
