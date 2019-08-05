from typing import Mapping, Any
from catalyst.dl.core import Runner


class ModelRunner(Runner):
    def predict_batch(self, batch: Mapping[str, Any]):
        # import pdb
        # pdb.set_trace()
        if 'cls_targets' in batch:
            output, cls_output = self.model(batch["images"])
            return {
                "logits": output,
                "cls_logits": cls_output
            }
        else:
            output = self.model(batch["images"])
            return {
                "logits": output
            }
