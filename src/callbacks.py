from catalyst.dl.core import Callback, RunnerState, MetricCallback
from catalyst.dl.callbacks import CriterionCallback
from catalyst.contrib.criterion import FocalLossBinary
from catalyst.dl.utils.criterion import accuracy
from catalyst.utils import get_activation_fn
import torch
import torch.nn as nn
import numpy as np
from typing import List
import logging
from slack_logger import SlackHandler, SlackFormatter


class SlackLogger(Callback):
    """
    Logger callback, translates state.metrics to console and text file
    """

    def __init__(self, url, channel):
        self.logger = None
        self.url = url
        self.channel = channel

    @staticmethod
    def _get_logger(url, channel):
        logger = logging.getLogger("metrics")
        logger.setLevel(logging.INFO)

        slackhandler = SlackHandler(
            username='logger',
            icon_emoji=':robot_face:',
            url=url,
            channel=channel
        )
        slackhandler.setLevel(logging.INFO)

        formater = SlackFormatter()
        slackhandler.setFormatter(formater)
        logger.addHandler(slackhandler)

        return logger

    def on_stage_start(self, state: RunnerState):
        self.logger = self._get_logger(self.url, self.channel)

    def on_stage_end(self, state):
        self.logger.handlers = []

    def on_epoch_end(self, state):
        pass
        # import pdb
        # pdb.set_trace()
        # self.logger.info("", extra={"state": state})


class LabelSmoothCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        loss = criterion(
            state.output[self.output_key],
            state.input[self.input_key]
        )
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class SmoothMixupCallback(LabelSmoothCriterionCallback):
    """
    Callback to do mixup augmentation.
    Paper: https://arxiv.org/abs/1710.09412
    Note:
        MixupCallback is inherited from CriterionCallback and
        does its work.
        You may not use them together.
    """

    def __init__(
        self,
        fields: List[str] = ("images",),
        alpha=0.5,
        on_train_only=True,
        **kwargs
    ):
        """
        Args:
            fields (List[str]): list of features which must be affected.
            alpha (float): beta distribution a=b parameters.
                Must be >=0. The more alpha closer to zero
                the less effect of the mixup.
            on_train_only (bool): Apply to train only.
                As the mixup use the proxy inputs, the targets are also proxy.
                We are not interested in them, are we?
                So, if on_train_only is True, use a standard output/metric
                for validation.
        """
        assert len(fields) > 0, \
            "At least one field for MixupCallback is required"
        assert alpha >= 0, "alpha must be>=0"

        super().__init__(**kwargs)

        self.on_train_only = on_train_only
        self.fields = fields
        self.alpha = alpha
        self.lam = 1
        self.index = None
        self.is_needed = True

    def on_loader_start(self, state: RunnerState):
        self.is_needed = not self.on_train_only or \
            state.loader_name.startswith("train")

    def on_batch_start(self, state: RunnerState):
        if not self.is_needed:
            return

        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)
        else:
            self.lam = 1

        self.index = torch.randperm(state.input[self.fields[0]].shape[0])
        self.index.to(state.device)

        for f in self.fields:
            state.input[f] = self.lam * state.input[f] + \
                (1 - self.lam) * state.input[f][self.index]

    def _compute_loss(self, state: RunnerState, criterion):
        if not self.is_needed:
            return super()._compute_loss(state, criterion)

        pred = state.output[self.output_key]
        y_a = state.input[self.input_key]
        y_b = state.input[self.input_key][self.index]

        loss = self.lam * criterion(pred, y_a) + \
            (1 - self.lam) * criterion(pred, y_b)
        return loss


class DSCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0,
        loss_weights: List[float] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.loss_weights = loss_weights

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        outputs = state.output[self.output_key]
        input = state.input[self.input_key]
        assert len(self.loss_weights) == len(outputs)
        loss = 0
        for i, output in enumerate(outputs):
            loss += criterion(output, input) * self.loss_weights[i]
        return loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        if state.loader_name.startswith("train"):
            criterion = state.get_key(
                key="criterion", inner_key=self.criterion_key
            )
        else:
            criterion = nn.CrossEntropyLoss()

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class DSAccuracyCallback(Callback):
    """
    Accuracy metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "acc",
        logit_names: List[str] = None,
    ):
        self.prefix = prefix
        self.metric_fn = accuracy
        self.input_key = input_key
        self.output_key = output_key
        self.logit_names = logit_names

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        assert len(outputs) == len(self.logit_names)

        batch_metrics = {}

        for logit_name, output in zip(self.logit_names, outputs):
            metric = self.metric_fn(output, targets)
            key = f"{self.prefix}_{logit_name}"
            batch_metrics[key] = metric[0]

        state.metrics.add_batch_value(metrics_dict=batch_metrics)


def dice_apex(outputs, targets, eps: float = 1e-7, activation: str = "Sigmoid"):
    """
    Computes the dice metric
        Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'sigmoid', 'softmax2d']
    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_fn(activation)
    targets = targets.float()

    outputs = activation_fn(outputs)

    batch_size = len(targets)

    with torch.no_grad():
        outputs = outputs.view(batch_size,-1)
        targets = targets.view(batch_size,-1)
        assert(outputs.shape==targets.shape)

        probability = outputs
        p = (probability>0.5).float()
        t = (targets>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum==0)
        pos_index = torch.nonzero(t_sum>=1)
        #print(len(neg_index), len(pos_index))


        dice_neg = (p_sum == 0).float()
        dice_pos = 2* (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice     = torch.cat([dice_pos,dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(),0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(),0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice


class DiceCallbackApex(MetricCallback):
    """
    Dice metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        eps: float = 1e-7,
        activation: str = "Sigmoid"
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=dice_apex,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            activation=activation
        )


class SIIMCriterionCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        cls_input_key: str = "cls_targets",
        output_key: str = "logits",
        cls_output_key: str = "cls_logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier
        self.cls_input_key = cls_input_key
        self.cls_output_key = cls_output_key

        self.cls_criterion = FocalLossBinary()

    def _add_loss_to_state(self, state: RunnerState, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state: RunnerState, criterion):
        seg_loss = criterion(
            state.output[self.output_key],
            state.input[self.input_key]
        )

        cls_loss = self.cls_criterion(
            state.output[self.cls_output_key],
            state.input[self.cls_input_key]
        )
        return seg_loss + cls_loss

    def on_stage_start(self, state: RunnerState):
        assert state.criterion is not None

    def on_batch_end(self, state: RunnerState):
        criterion = state.get_key(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)