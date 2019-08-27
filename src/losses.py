from functools import partial
from catalyst.dl.utils import criterion
from catalyst.utils import get_activation_fn
from catalyst.contrib.criterion import LovaszLossBinary, FocalLossBinary

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    Computes the dice metric

    Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)
    targets = targets.float()

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice


def soft_dice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid",
    weight=[0.2, 0.8]
):
    """
    Computes the dice metric

    Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)
    targets = targets.float()

    batch_size = len(outputs)
    outputs = outputs.view(batch_size, -1)
    targets = targets.view(batch_size, -1)

    p = outputs.view(batch_size, -1)
    t = targets.view(batch_size, -1)
    w = targets.detach()
    w = w*(weight[1]-weight[0])+weight[0]

    p = w*(p*2-1)
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union = (p * p).sum(-1) + (t * t).sum(-1)
    dice = 1 - 2*intersection/union

    loss = dice
    return loss.mean()


class DiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()

        self.loss_fn = partial(
            dice,
            eps=eps,
            threshold=threshold,
            activation=activation)

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return 1 - dice


class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()

        self.loss_fn = partial(
            soft_dice,
            eps=eps,
            threshold=threshold,
            activation=activation)

    def forward(self, logits, targets):
        dice = self.loss_fn(logits, targets)
        return dice


class WeightedBCE(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        logit = outputs.view(-1)
        truth = targets.view(-1)
        assert (logit.shape == truth.shape)

        loss = self.bce_loss(logit, truth)
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25 * pos * loss / pos_weight + 0.75 * neg * loss / neg_weight).sum()

        return loss


class BCEDiceLossApex(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = 0.1 * dice + bce * 0.9
        return loss


class WeightedBCEDiceLossApex(nn.Module):
    def __init__(
        self,
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid"
    ):
        super().__init__()
        self.bce_loss = WeightedBCE()
        self.dice_loss = SoftDiceLoss(
            eps=eps,
            threshold=threshold,
            activation=activation
        )

    def forward(self, outputs, targets):
        dice = self.dice_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = 0.1 * dice + bce * 0.9
        return loss


class BCEFocalLossApex(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLossBinary()

    def forward(self, outputs, targets):
        focal = self.focal_loss(outputs, targets)
        bce = self.bce_loss(outputs, targets)
        loss = focal + bce
        return loss