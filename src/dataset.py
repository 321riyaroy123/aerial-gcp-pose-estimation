import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224

SHAPE_CLASSES = ["Cross", "Square", "L-Shape"]
SHAPE2IDX = {s: i for i, s in enumerate(SHAPE_CLASSES)}


train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=15,
        border_mode=0,
        p=0.6
    ),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


class GCPDataset(Dataset):

    def __init__(self, samples, base_dir, transform=None):
        self.samples = samples
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        s = self.samples[idx]

        img_path = os.path.join(self.base_dir, s["rel_path"])
        image = np.array(Image.open(img_path).convert("RGB"))

        orig_w = s.get("orig_w", image.shape[1])
        orig_h = s.get("orig_h", image.shape[0])

        x_norm = s.get("x_norm", 0.5)
        y_norm = s.get("y_norm", 0.5)

        if self.transform:

            h, w = image.shape[:2]
            kp = [(x_norm * w, y_norm * h)]

            result = self.transform(image=image, keypoints=kp)

            image = result["image"]
            new_kp = result["keypoints"]

            if len(new_kp) > 0:
                nx, ny = new_kp[0]

                x_norm = np.clip(nx / IMG_SIZE, 0.0, 1.0)
                y_norm = np.clip(ny / IMG_SIZE, 0.0, 1.0)

        coords = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        shape_idx = torch.tensor(s.get("shape_idx", -1), dtype=torch.long)

        return image, coords, shape_idx, s["rel_path"], orig_w, orig_h


def load_annotations(annot_file, train_dir):

    with open(annot_file) as f:
        data = json.load(f)

    label_lookup = {k.replace("\\","/"): v for k,v in data.items()}

    samples = []

    for img_path in sorted(Path(train_dir).rglob("*.JPG")):

        rel = str(img_path.relative_to(train_dir)).replace("\\","/")

        if rel not in label_lookup:
            continue

        info = label_lookup[rel]

        if "verified_shape" not in info:
            continue

        shape = info["verified_shape"]

        if shape not in SHAPE2IDX:
            continue

        with Image.open(img_path) as im:
            orig_w, orig_h = im.size

        samples.append({
            "rel_path": rel,
            "x_norm": info["mark"]["x"] / orig_w,
            "y_norm": info["mark"]["y"] / orig_h,
            "shape_idx": SHAPE2IDX[shape],
            "orig_w": orig_w,
            "orig_h": orig_h
        })

    return samples


def load_test_samples(test_dir):

    samples = []

    for img_path in sorted(Path(test_dir).rglob("*.JPG")):

        rel = str(img_path.relative_to(test_dir)).replace("\\","/")

        with Image.open(img_path) as im:
            orig_w, orig_h = im.size

        samples.append({
            "rel_path": rel,
            "orig_w": orig_w,
            "orig_h": orig_h
        })

    return samples
