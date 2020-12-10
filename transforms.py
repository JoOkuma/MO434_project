# code from: https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
# modified by jordao bragantini

import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, targets = None):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            if targets:
                for target in targets:
                    boxes = target["boxes"]
                    boxes[0] = width - boxes[0]
                    boxes[2] = width - boxes[2]
                    x0 = min(boxes[0], boxes[2])
                    x1 = max(boxes[0], boxes[2])
                    target["boxes"] = torch.tensor([x0, boxes[1], x1, boxes[3]])
                    if "masks" in target:
                        target["masks"] = target["masks"].flip(-1)
                    if "keypoints" in target:
                        keypoints = target["keypoints"]
                        keypoints = _flip_coco_person_keypoints(keypoints, width)
                        target["keypoints"] = keypoints
        return image, targets


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

