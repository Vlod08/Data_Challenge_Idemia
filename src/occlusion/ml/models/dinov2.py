from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DINOV2_VARIANT = Literal[
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
    "dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg",
]
FinetuneMode = Literal["frozen", "lora", "full"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default LoRA targets: attention (qkv + output projection).
DEFAULT_LORA_TARGETS = ("qkv", "proj")


def default_dinov2_transform(img_size: int = 224):
    """Deterministic transform (validation / test / embedding extraction)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def train_dinov2_transform(img_size: int = 224, augment: bool = True):
    """Training transform with label-preserving augmentations only: horizontal
    flip + photometric jitter. No crop/erasing, which would change the visible
    face area (hence the occlusion ground-truth)."""
    tfs = [transforms.Resize((img_size, img_size),
                             interpolation=InterpolationMode.BICUBIC)]
    if augment:
        tfs += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
        ]
    tfs += [transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(tfs)


class Dinov2Backbone(nn.Module):
    """Frozen backbone used for embedding extraction (fast head-only pipeline)."""

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
            self.model.eval()
        return self

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)


def _apply_lora(model: nn.Module, r: int, alpha: int, dropout: float,
                targets: Sequence[str]) -> nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft is required for LoRA mode (`uv add peft`).") from exc
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=list(targets), bias="none",
    )
    return get_peft_model(model, cfg)


class Dinov2Regressor(nn.Module):
    """DINOv2 + a single-output head. head_activation="sigmoid" bounds the output
    to [0, 1]; "linear" returns the raw value (clamped to [0, 1] at inference).
    The sigmoid output doubles as P(class=1) for the gender classifier.

    finetune_mode:
      - "frozen": backbone frozen, only the head learns (no_grad forward).
      - "lora"  : backbone frozen + trainable LoRA adapters on attention.
      - "full"  : the whole backbone is fine-tuned.
    """

    def __init__(
        self,
        model_name: DINOV2_VARIANT = "dinov2_vits14",
        finetune_mode: FinetuneMode = "frozen",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        head_activation: str = "sigmoid",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
    ):
        super().__init__()
        self.model_name = model_name
        self.finetune_mode = finetune_mode
        self.head_activation = head_activation

        backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.embed_dim = backbone.embed_dim

        if finetune_mode == "frozen":
            for p in backbone.parameters():
                p.requires_grad = False
        elif finetune_mode == "full":
            pass  # everything trainable
        elif finetune_mode == "lora":
            for p in backbone.parameters():
                p.requires_grad = False
            backbone = _apply_lora(backbone, lora_r, lora_alpha,
                                   lora_dropout, lora_targets)
        else:
            raise ValueError(f"unknown finetune_mode: {finetune_mode}")

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.finetune_mode == "frozen":
            self.backbone.eval()  # no dropout/stochastic-depth on the frozen backbone
        return self

    def _features(self, x):
        if self.finetune_mode == "frozen":
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)

    def forward(self, x):
        feat = self._features(x)
        out = self.head(feat).squeeze(-1)
        # "linear" returns the raw output (clamped to [0,1] only at inference,
        # see engine.evaluate / predict_catalog) so gradients are not saturated.
        return torch.sigmoid(out) if self.head_activation == "sigmoid" else out

    def param_groups(self, backbone_lr: float, head_lr: float,
                     weight_decay: float = 0.0):
        """Discriminative backbone/head LRs. No weight decay on 1-D params
        (LayerNorm weights, biases), as is standard for ViT fine-tuning."""
        def split(module):
            decay, no_decay = [], []
            for p in module.parameters():
                if p.requires_grad:
                    (no_decay if p.ndim <= 1 else decay).append(p)
            return decay, no_decay

        groups = []
        for module, lr in ((self.head, head_lr), (self.backbone, backbone_lr)):
            decay, no_decay = split(module)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
        return groups

    def trainable_parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "trainable_pct": round(100 * trainable / max(total, 1), 3)}
