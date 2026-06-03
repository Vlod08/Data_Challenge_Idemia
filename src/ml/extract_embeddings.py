import os
import argparse
import torch
from torch.utils.data import DataLoader

from src import config
from src import utils
from src.ml.core.dataset import Image_Dataset
from src.ml.models.dinov2 import Dinov2Backbone, default_dinov2_transform

PATHS = config.PATHS


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=224)
    args = p.parse_args()

    device = get_device()
    print(f"device: {device}")
    transform = default_dinov2_transform(img_size=args.img_size)
    backbone = Dinov2Backbone(args.model_type, freeze=True).to(device)

    out_dir = os.path.join(PATHS.artifacts_dir, "embeddings", args.model_type)
    os.makedirs(out_dir, exist_ok=True)

    train_cat = utils.load_catalog_csv(PATHS.train_catalog_path)
    train_loader = DataLoader(
        Image_Dataset(catalog=train_cat, mode="Train", transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("extracting train embeddings...")
    torch.save(extract(backbone, train_loader, device, True),
               os.path.join(out_dir, "train.pt"))

    test_cat = utils.load_catalog_csv(PATHS.test_catalog_path)
    test_loader = DataLoader(
        Image_Dataset(catalog=test_cat, mode="Test", transform=transform),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print("extracting test embeddings...")
    torch.save(extract(backbone, test_loader, device, False),
               os.path.join(out_dir, "test.pt"))

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
