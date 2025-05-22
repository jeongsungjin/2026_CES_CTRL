# project_bev/models/custom_bev_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CustomBEVBackbone(nn.Module):
    # """ResNet-34 + FPN → 256×25×25 BEV Feature."""
    def __init__(self, out_ch: int = 256):
        super().__init__()
        res = torchvision.models.resnet34(weights='IMAGENET1K_V1')

        # ── Stem (stride 1) ───────────────────────────────
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = res.bn1
        self.relu  = nn.ReLU(True)

        # ── ResNet blocks ────────────────────────────────
        self.layer1, self.layer2 = res.layer1, res.layer2   # stride 1
        self.layer3, self.layer4 = res.layer3, res.layer4   # stride 2

        # ── FPN 1×1 convs ────────────────────────────────
        self.l2_256 = nn.Conv2d(128, out_ch, 1)
        self.l3_256 = nn.Conv2d(256, out_ch, 1)
        self.l4_256 = nn.Conv2d(512, out_ch, 1)
        self.out_conv = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        # learnable BEV positional embedding
        self.pos = nn.Parameter(torch.randn(1, out_ch, 25, 25))

    # ----------------------------------------------------
    def forward(self, x):                     # [B,3,200,200]
        x  = self.relu(self.bn1(self.conv1(x)))
        x  = self.layer1(x)                   # 200×200
        c2 = self.layer2(x)                   # 200×200
        c3 = self.layer3(c2)                  # 100×100
        c4 = self.layer4(c3)                  #  50×50

        p2 = F.interpolate(self.l2_256(c2), (25, 25))
        p3 = F.interpolate(self.l3_256(c3), (25, 25))
        p4 = self.l4_256(c4)                  # already 25×25
        bev = self.out_conv(p2 + p3 + p4)     # [B,256,25,25]

        return bev, self.pos
