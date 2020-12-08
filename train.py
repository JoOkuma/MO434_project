import torch
import torchvision
import pytorch_lightning as pl

from config import Config
from model import get_segmentation_model
from litmodel import LitModel
import transforms as T


def get_transform(train: bool):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    train_loader = torchvision.datasets.CocoDetection(
            Config.images_dir(),
            Config.annot_path(),
            transforms=get_transform(True),
    )

    trainer = pl.Trainer(gpus=[0])
    model = LitModel(get_segmentation_model(10))  # FIXME number of classes is not correct, just testing

    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()

