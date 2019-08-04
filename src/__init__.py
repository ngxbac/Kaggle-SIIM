# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from losses import *
from callbacks import *
from optimizers import *
from schedulers import *


# Register models
registry.Model(Unet)
registry.Model(model34_DeepSupervion)
registry.Model(HyperUnet)

# Register callbacks
registry.Callback(LabelSmoothCriterionCallback)
registry.Callback(SmoothMixupCallback)
registry.Callback(DSAccuracyCallback)
registry.Callback(DSCriterionCallback)
registry.Callback(SlackLogger)
registry.Callback(DiceCallbackApex)

# Register criterions
registry.Criterion(LabelSmoothingCrossEntropy)

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)

registry.Scheduler(CyclicLRFix)