import os
import torch
import torchvision
from typing import Dict, Optional
from PIL import Image
import csv
import pycocotools


class TACODataset(torchvision.datasets.CocoDetection):
    def __init__(self, class_map: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        
        self.class_map = class_map
        self.n_classes = -1
        if self.class_map:
            self.n_classes = len(set(self.class_map.values()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        targets = self.preprocess(img, coco.loadAnns(ann_ids))

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)

        return img, targets

    def preprocess(self, img, targets):
        new_targets = []
        for target in targets:
            if 'bbox' not in target:
                print(target)
                continue
            x, y, w, h = target['bbox']
            target['boxes'] = torch.Tensor([x, y, x + w, y + h])
            if w < 10 or h < 10:
                # TOO SMALL REMOVE
                continue
            del target['bbox']
            h, w = img.height, img.width
            target['masks'] = torch.tensor(_annToMask(target, h, w))
            cat_id = target['category_id']
            if self.class_map:
                cat_id = self.class_map[cat_id]
            target['labels'] = torch.tensor(cat_id)
            new_targets.append(target)

        return new_targets


def process_csv(csvpath: str) -> Dict:
    str2id = {}
    class2id = {}
    with open(csvpath) as f:
        reader = csv.reader(f)
        count = 0
        for i, row in enumerate(reader):
            if row[1] in str2id:
                class2id[i] = str2id[row[1]]
            else:
                str2id[row[1]] = count
                class2id[i] = count
                count += 1
    return class2id
  

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

