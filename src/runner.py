from typing import Mapping, Any
from catalyst.dl.core import Runner


class ModelRunner(Runner):
    def predict_batch(self, batch: Mapping[str, Any]):
        # import pdb
        # pdb.set_trace()
        pred = self.model(batch["images"])
        if len(pred) == 2:
            output, cls_output = pred
            return {
                "logits": output,
                "cls_logits": cls_output
            }
        else:
            output = pred
            return {
                "logits": output
            }