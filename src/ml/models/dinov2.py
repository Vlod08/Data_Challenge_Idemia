import torch
import torch.nn as nn

from typing import Literal
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


DINOV2_VARIANT = Literal[
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vits14_reg",
    "dinov2_vitb14_reg",
    "dinov2_vitl14_reg",
    "dinov2_vitg14_reg",
]


def default_dinov2_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


class Dinov2(nn.Module):
    def __init__(
        self,
        model_name: DINOV2_VARIANT = "dinov2_vits14",
        device: str = "cuda",
        transform=None,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.transform = transform 

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, X): 
        # X : B, C, H, W
        
        X = X.to(self.device)
        if self.transform is not None:
            X = self.transform(X)
        X = X.to(self.device)

        return self.model(X)