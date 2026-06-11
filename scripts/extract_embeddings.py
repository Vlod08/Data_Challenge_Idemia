"""Extract frozen DINOv2 embeddings (fast head-only pipeline).

  uv run python scripts/extract_embeddings.py --model-type dinov2_vits14
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from occlusion import utils
from occlusion.config import PATHS
from occlusion.ml.core import engine
from occlusion.ml.core.dataset import Image_Dataset
from occlusion.ml.models.dinov2 import Dinov2Backbone, default_dinov2_transform


@torch.no_grad()
def extract(backbone, loader, device, has_labels):
    backbone.eval()
    embs, occs, genders, filenames = [], [], [], []
    nb = len(loader)
    for i, batch in enumerate(loader):
        if has_labels:
            images, occ, gender = batch
            occs.append(occ)
            genders.append(gender)
        else:
            images, fns = batch
            filenames.extend(fns)
        embs.append(backbone(images.to(device)).cpu())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{nb} batches", flush=True)
    out = {"embeddings": torch.cat(embs)}
    if has_labels:
        out["occ"] = torch.cat(occs)
        out["gender"] = torch.cat(genders)
    else:
        out["filenames"] = filenames
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", default="dinov2_vits14")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    device = engine.get_device()
    print(f"device: {device}")
    transform = default_dinov2_transform(img_size=args.img_size)
    backbone = Dinov2Backbone(args.model_type, freeze=True).to(device)

    out_dir = os.path.join(PATHS.artifacts_dir, "embeddings", args.model_type)
    os.makedirs(out_dir, exist_ok=True)

    for split, cat_path, has_labels in (
        ("train", PATHS.train_catalog_path, True),
        ("test", PATHS.test_catalog_path, False),
    ):
        cat = utils.load_catalog_csv(cat_path)
        loader = DataLoader(
            Image_Dataset(catalog=cat, mode="Train" if has_labels else "Test",
                          transform=transform),
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"extracting {split} embeddings...")
        torch.save(extract(backbone, loader, device, has_labels),
                   os.path.join(out_dir, f"{split}.pt"))

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
