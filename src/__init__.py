# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *
from optimizers import *
from schedulers import *
# from segmentation_models_pytorch import Linknet


# Register models
registry.Model(Linknet)
registry.Model(FPN)
registry.Model(PSPNet)
registry.Model(Unet)
registry.Model(UnetIBN)
# registry.Model(model34_DeepSupervion)
# registry.Model(HyperUnet)
registry.Model(UnetSCSE)
registry.Model(Res34Unetv4)
registry.Model(UnetMix)
registry.Model(UnetOC)
registry.Model(LadderNetv6)

# Register callbacks
registry.Callback(LabelSmoothCriterionCallback)
registry.Callback(SmoothMixupCallback)
registry.Callback(DSAccuracyCallback)
registry.Callback(DSCriterionCallback)
registry.Callback(SlackLogger)
registry.Callback(DiceCallbackApex)
registry.Callback(SIIMCriterionCallback)

# Register criterions
registry.Criterion(LabelSmoothingCrossEntropy)
registry.Criterion(BCEDiceLossApex)
registry.Criterion(BCEFocalLossApex)
registry.Criterion(WeightedBCE)
registry.Criterion(WeightedBCEDiceLossApex)

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)

registry.Scheduler(CyclicLRFix)