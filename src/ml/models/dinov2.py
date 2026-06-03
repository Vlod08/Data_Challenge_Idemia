import torch
import torch.nn as nn
from typing import Literal
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DINOV2_VARIANT = Literal[
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
    "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_dinov2_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class Dinov2Backbone(nn.Module):
    def __init__(self, model_name: DINOV2_VARIANT = "dinov2_vits14", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = self.model.embed_dim
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()  # garde le backbone gelé en eval même en mode train
        return self

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)


class Dinov2Regressor(nn.Module):
    def __init__(self, model_name: DINOV2_VARIANT = "dinov2_vits14",
                 freeze_backbone: bool = True, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = Dinov2Backbone(model_name, freeze=freeze_backbone)
        d = self.backbone.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return torch.sigmoid(self.head(feat).squeeze(-1))  # borné dans [0, 1]
