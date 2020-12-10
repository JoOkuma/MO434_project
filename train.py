import torch
import torchvision
import pytorch_lightning as pl
from copy import copy

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
        # transforms.append(T.Resize())
    return T.Compose(transforms)


def get_splits(dataset, val_size = 150, test_size = 250):
    train_size = len(dataset) - val_size - test_size
    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )


def main():
    ### loading data ###
    class_map = process_csv(Config.class4_config())

    dataset = TACODataset(
        class_map,
        root=Config.images_dir(),
        annFile=Config.annot_path(),
        transforms=get_transform(False),
    )

    train_ds, val_ds, test_ds = get_splits(dataset)

    # hack to use different transforms
    train_ds.dataset = copy(dataset)
    train_ds.dataset.transforms = get_transform(True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True,
        pin_memory=True, num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1,
        pin_memory=True, num_workers=4,
    )
    ### loading data ###


    trainer = pl.Trainer(
        gpus=[4, 5, 6, 7],
        accumulate_grad_batches=2,
        # sync_batchnorm=True,
        terminate_on_nan=True,
        accelerator='ddp',
    )

    model = LitModel(get_segmentation_model(dataset.n_classes), dataset)

    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()

