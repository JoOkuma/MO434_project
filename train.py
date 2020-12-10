import torch
import torchvision
import pytorch_lightning as pl

from config import Config
from taco import TACODataset, process_csv
from model import get_segmentation_model
from litmodel import LitModel
import transforms as T


def get_transform(train: bool):
    transforms = [
        T.ToTensor(),
    ]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    class_map = process_csv(Config.class4_config())

    dataset = TACODataset(
        class_map,
        root=Config.images_dir(),
        annFile=Config.annot_path(),
        transforms=get_transform(True),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        pin_memory=True, num_workers=4,
    )

    trainer = pl.Trainer(
        gpus=[0],
        accumulate_grad_batches=2,
        # sync_batchnorm=True,
        terminate_on_nan=True,
    )

    model = LitModel(get_segmentation_model(dataset.n_classes))

    trainer.fit(model, train_loader)


if __name__ == '__main__':
    main()

