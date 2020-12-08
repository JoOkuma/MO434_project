import torch
import torch.nn as nn
import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss_dict = self.model(x, y)
        return sum(loss for loss in loss_dict.values())

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

