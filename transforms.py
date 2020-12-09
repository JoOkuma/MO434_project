# code from: https://github.com/pytorch/vision/blob/master/references/detection/transforms.py
# modified by jordao bragantini

import random
import torch

from torchvision.transforms import functional as F
import pycocotools


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


def _annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = pycocotools.mask.frPyObjects(segm, height, width)
        rle = pycocotools.mask.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = pycocotools.mask.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def _annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = _annToRLE(ann, height, width)
    m = pycocotools.mask.decode(rle)
    return m


class PreProcess:
    def __call__(self, image, targets):

        for target in targets:
            x, y, w, h = target['bbox']
            target['boxes'] = torch.Tensor([x, y, x + w, y + h])
            del target['bbox']
            h, w = image.shape[1:]
            target['masks'] = torch.tensor(_annToMask(target, h, w))
            target['labels'] = torch.tensor(target['category_id'])

        return image, targets

