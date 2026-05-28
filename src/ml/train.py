import torch
import torch.nn as nn
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

from src import config
from src import utils
from src.ml.core.dataset import Image_Dataset

PATHS = config.PATHS


def main():

    args = config.parse_args()
    MODEL_TYPE = args.model_type
    MODELS_DIR = args.output_dir
    EXPERIMENT_NAME = args.experiment_name
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    EPOCHS = args.epochs
    NUM_WORKERS = args.num_workers
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    VAL_SPLIT = args.val_split
    IN_CHANNELS = args.in_channels
    RANDOM_SEED = args.random_seed
    RESUME_CHECKPOINT = args.resume_checkpoint
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    EXP_DIR = os.path.join(MODELS_DIR, EXPERIMENT_NAME)
    RUNS_DIR = os.path.join(EXP_DIR, "tensorboard")
    VIZ_OUTPUT_DIR = os.path.join(EXP_DIR, "visualizations")
    BEST_MODEL_PATH = os.path.join(EXP_DIR, "model_val_best.pth")
    BEST_SUB_MODEL_PATH = os.path.join(EXP_DIR, "model_sub_best.pth")
    LAST_MODEL_PATH = os.path.join(EXP_DIR, "model_last.pth")
    LAST_CHECKPOINT_PATH = os.path.join(EXP_DIR, "checkpoint_last.pt")
    LOSS_CURVE_PATH = os.path.join(EXP_DIR, "loss_curve.png")

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(VIZ_OUTPUT_DIR, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED) 
    random.seed(RANDOM_SEED)

    # Print the parsed arguments
    utils.print_args(args)

    print("\n--- 1. Data Setup ---\n")


    catalog = utils.load_catalog_csv(PATHS.train_catalog_path)
    train, val = train_test_split(
        catalog,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
    )

    dt_full = Image_Dataset(catalog=catalog, img_size=tuple(IMG_SIZE))
    dt_train = Image_Dataset(catalog=train, img_size=tuple(IMG_SIZE))
    dt_val = Image_Dataset(catalog=val, img_size=tuple(IMG_SIZE))

    print(len(dt_full))
    print(len(dt_train))
    print(len(dt_val))




if __name__ == '__main__':
    main() 

