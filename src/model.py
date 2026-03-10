import torch
import torch.nn as nn
import torchvision.models as models


class GCPModel(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):

        feats = self.backbone(x)

        coords = self.reg_head(feats)
        logits = self.cls_head(feats)

        return coords, logits
