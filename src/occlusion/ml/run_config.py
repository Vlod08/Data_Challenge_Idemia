"""Training configuration: a single dataclass loaded from a YAML file and
optionally overridden on the command line. One YAML per experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class TrainConfig:
    # run
    experiment_name: str = "run"
    seed: int = 42
    device: str = "auto"               # auto | cuda | mps | cpu

    # model
    model_type: str = "dinov2_vitb14_reg"
    finetune_mode: str = "lora"        # frozen | lora | full
    hidden_dim: int = 256
    head_dropout: float = 0.2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: ["qkv", "proj"])

    # data
    img_size: int = 224
    val_split: float = 0.2
    augment: bool = True
    sampler: str = "gender"            # none | gender | gender_occ
    occ_power: float = 0.5             # tail oversampling (gender_occ only)

    # optimisation
    loss: str = "balanced"            # balanced | wmse
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 8
    head_lr: float = 1e-3
    backbone_lr: float = 2e-4         # LoRA adapter LR; lower it (~1e-5) for full FT
    weight_decay: float = 1e-4        # applied to 2-D weights only (not norms/bias)
    scheduler: str = "cosine"          # cosine | constant | none
    warmup_frac: float = 0.05
    grad_accum: int = 1
    clip_grad: float = 0.0
    amp: bool = True

    # inference
    tta: bool = False

    # debug: tiny local run (subsampled data, few epochs)
    debug: bool = False

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


_FIELD_TYPES = {f.name: f.type for f in fields(TrainConfig)}


def _cast(name: str, value: Any) -> Any:
    """Cast a CLI string override to the field's type (best effort)."""
    if not isinstance(value, str):
        return value
    ftype = _FIELD_TYPES.get(name, str)
    if ftype is bool or ftype == "bool":
        return value.lower() in ("1", "true", "yes")
    if ftype is int or ftype == "int":
        return int(value)
    if ftype is float or ftype == "float":
        return float(value)
    if "list" in str(ftype):
        return value.split(",")
    return value


def load_config(yaml_path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> TrainConfig:
    data: dict[str, Any] = {}
    if yaml_path is not None:
        data = yaml.safe_load(Path(yaml_path).read_text()) or {}
    for key, value in (overrides or {}).items():
        data[key] = _cast(key, value)

    known = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainConfig(**data)


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Parse `key=value` strings from the CLI into a dict."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got: {pair}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    return asdict(cfg)
