import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, dataset):
        super().__init__()

        self.model = model
        self.dataset = dataset

    def _get_coco_eval(self):
        coco = get_coco_api_from_dataset(self.dataset)
        iou_types = ['segm', 'bbox']
        return CocoEvaluator(coco, iou_types)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache() 
        x, y = batch
        loss_dict = self.model(x, y)
        return sum(loss for loss in loss_dict.values())

    def validation_step(self, batch, batch_idx):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache() 
        x, targets = batch
        outputs = self.model(x)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_eval.update(res)

    def on_validation_epoch_start(self):
        self.n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.coco_eval = self._get_coco_eval()

    def validation_epoch_end(self, validation_steps_outputs):
        self.coco_eval.synchronize_between_processes()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()
        torch.set_num_threads(self.n_threads)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.SGD(params,
                                    lr=0.005,
                                    momentum=0.9,
                                    weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
        return [optimizer], [scheduler]

