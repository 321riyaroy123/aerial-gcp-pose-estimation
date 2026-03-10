import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import (
    GCPDataset,
    load_annotations,
    train_transform,
    val_transform,
    SHAPE_CLASSES
)

from model import GCPModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = "train_dataset"
ANNOT_FILE = os.path.join(TRAIN_DIR,"gcp_marks.json")
WEIGHTS_PATH = "weights/best_model.pth"


def train():

    samples = load_annotations(ANNOT_FILE, TRAIN_DIR)

    train_s, val_s = train_test_split(
        samples,
        test_size=0.15,
        random_state=42,
        stratify=[s["shape_idx"] for s in samples]
    )

    train_ds = GCPDataset(train_s, TRAIN_DIR, train_transform)
    val_ds = GCPDataset(val_s, TRAIN_DIR, val_transform)

    train_loader = DataLoader(train_ds,batch_size=16,shuffle=True)
    val_loader = DataLoader(val_ds,batch_size=16,shuffle=False)

    model = GCPModel().to(device)

    optimizer = optim.AdamW(model.parameters(),lr=1e-4)

    loc_loss = nn.SmoothL1Loss()
    ce_loss = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(10):

        model.train()

        for imgs,coords,labels,_,_,_ in tqdm(train_loader):

            imgs = imgs.to(device)
            coords = coords.to(device)
            labels = labels.to(device)

            pc,pl = model(imgs)
            pc = torch.sigmoid(pc)

            loss = 50*loc_loss(pc,coords)+ce_loss(pl,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch finished")

        torch.save(model.state_dict(),WEIGHTS_PATH)


if __name__ == "__main__":
    train()
