import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torch
from dataset.tfs import get_polyp_transform
import cv2
from pathlib import Path
import re
from typing import Literal

RETINA_ROOT_DIR = Path("/dhc/dsets/REFUGE/REFUGE")


class RetinaDataset(data.Dataset):

    def __init__(
        self,
        image_root,
        target: Literal["disc", "cup"],
        trainsize=352,
        augmentations=None,
        train=True,
        sam_trans=None,
    ):
        self.trainsize = trainsize
        self.augmentations = augmentations
        # print(self.augmentations)

        images_and_masks = [
            (
                str(image_root / dir / f"{dir}.jpg"),
                str(image_root / dir / f"{dir}_seg_{target}_1.png"),
            )
            for dir in os.listdir(image_root)
            if re.search("\d\d\d\d", dir) is not None
        ]

        self.images = [img for img, _ in images_and_masks]
        self.gts = [mask for _, mask in images_and_masks]

        self.filter_files()
        self.size = len(self.images)
        self.train = train
        self.sam_trans = sam_trans

    def __getitem__(self, index):
        image = self.cv2_loader(self.images[index], is_mask=False)
        gt = self.cv2_loader(self.gts[index], is_mask=True)
        # image = self.rgb_loader(self.images[index])
        # gt = self.binary_loader(self.gts[index])
        img, mask = self.augmentations(image, gt)
        # mask[mask >= 128] = 255
        # mask[mask < 128] = 0
        # mask[mask == 255] = 1
        # mask = mask.squeeze()
        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(
            img
        ), self.sam_trans.apply_image_torch(mask)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])
        return (
            self.sam_trans.preprocess(img),
            self.sam_trans.preprocess(mask),
            torch.Tensor(original_size),
            torch.Tensor(image_size),
        )
        # return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def binary_loader(self, path):
        # with open(path, 'rb') as f:
        # img = Image.open(f)
        # return img.convert('1')
        img = cv2.imread(path, 0)
        return img

    def cv2_loader(self, path, is_mask):
        if is_mask:
            img = cv2.imread(path, 0)
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        # return 32
        return self.size


def get_retina_dataset(args, sam_trans=None, target=Literal["disc", "cup"]):
    transform_train, transform_test = get_polyp_transform()
    image_root = RETINA_ROOT_DIR / "Training-400"
    ds_train = RetinaDataset(
        image_root, target, augmentations=transform_train, sam_trans=sam_trans
    )
    image_root = RETINA_ROOT_DIR / "Test-400"
    ds_test = RetinaDataset(
        image_root,
        target,
        train=False,
        augmentations=transform_test,
        sam_trans=sam_trans,
    )
    return ds_train, ds_test
