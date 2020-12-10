# code from: https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
# heavily modified by jordao bragantini

import random
import torch
import numpy as np
from PIL import Image

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

class Resize:
    def __init__(self, minimum_dim=800):
        self.minimum_dim = minimum_dim

    def __call__(self, image, targets):
        h, w = image.shape[1:]
        if h > w:
            new_w = self.minimum_dim
            new_h = int(round(h * new_w / w))
        else:
            new_h = self.minimum_dim
            new_w = int(round(w * new_h / h))

        size = (new_h, new_w)
        image = F.resize(image, size)

        new_targets = []
        for target in targets:
            target['masks'] = F.resize(target['masks'].unsqueeze(0), size,
                                       interpolation=Image.NEAREST).squeeze()

            if target['masks'].sum() > 0:
                pos = np.where(target['masks'].numpy())
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
            else:
                xmin, ymin, xmax, ymax = 0, 0, 0, 0

            xmin = max(0, xmin - 2)
            xmax = min(new_w, xmax + 2)
            ymin = max(0, ymin - 2)
            ymax = min(new_h, ymax + 2)

            target['boxes'] = torch.Tensor([xmin, ymin, xmax, ymax])
            new_targets.append(target)

        return image, new_targets


