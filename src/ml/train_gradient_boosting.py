import json
import os
import pickle
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src import config
from src.ml.core.metrics import challenge_score


PATHS = config.PATHS


def build_xgboost(
    n_estimators: int,
    learning_rate: float,
    random_state: int,
    objective: str = "reg:squarederror",
) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        objective=objective,
        n_jobs=-1,
        tree_method="hist",
    )


def build_lgbm(
    n_estimators: int,
    learning_rate: float,
    random_state: int,
) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def as_score_tensors(preds: np.ndarray, targets: np.ndarray, genders: np.ndarray):
    preds = np.clip(preds, 0.0, 1.0)
    return (
        torch.from_numpy(preds.astype(np.float32)),
        torch.from_numpy(targets.astype(np.float32)),
        torch.from_numpy(genders.astype(np.float32)),
    )


def model_predict(model, name: str, X: np.ndarray, iteration: int | None = None) -> np.ndarray:
    if iteration is None:
        return model.predict(X)
    if name == "xgboost":
        return model.predict(X, iteration_range=(0, iteration))
    if name == "lightgbm":
        return model.booster_.predict(X, num_iteration=iteration)
    raise ValueError(f"Unknown model name: {name}")


def evaluate_predictions(
    model_name: str,
    iteration: int,
    preds: np.ndarray,
    y_val: np.ndarray,
    gen_val: np.ndarray,
) -> dict[str, float | int | str]:
    pred_t, y_t, gen_t = as_score_tensors(preds, y_val, gen_val)
    metric = challenge_score(pred_t, y_t, gen_t)
    return {
        "model": model_name,
        "iteration": iteration,
        "rmse": mean_squared_error(y_val, np.clip(preds, 0.0, 1.0)) ** 0.5,
        "r2": r2_score(y_val, np.clip(preds, 0.0, 1.0)),
        "val_score": metric["score"],
        "err_g0": metric["err_g0"],
        "err_g1": metric["err_g1"],
    }


def build_iteration_history(
    model,
    model_name: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    gen_val: np.ndarray,
    n_estimators: int,
) -> pd.DataFrame:
    rows = []
    print_every = max(1, n_estimators // 10)
    for iteration in range(1, n_estimators + 1):
        preds = model_predict(model, model_name, X_val, iteration)
        row = evaluate_predictions(model_name, iteration, preds, y_val, gen_val)
        rows.append(row)
        if iteration == 1 or iteration % print_every == 0 or iteration == n_estimators:
            print(
                f"{model_name} iteration {iteration:03d} "
                f"| val_score {row['val_score']:.5f} "
                f"| g0 {row['err_g0']:.5f} | g1 {row['err_g1']:.5f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def save_model(model, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(model, file)


def save_curve(history: pd.DataFrame, exp_dir: Path) -> None:
    for model_name, model_history in history.groupby("model"):
        plt.plot(
            model_history["iteration"],
            model_history["val_score"],
            label=model_name,
        )
    plt.xlabel("iteration")
    plt.ylabel("validation challenge score")
    plt.legend()
    plt.savefig(exp_dir / "boosting_val_score.png")
    plt.close()


def main():
    args = config.parse_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    emb_dir = PATHS.artifacts_dir / "embeddings" / args.model_type
    train_blob = torch.load(emb_dir / "train.pt")
    X = to_numpy(train_blob["embeddings"])
    y = to_numpy(train_blob["occ"]).reshape(-1)
    gen = to_numpy(train_blob["gender"]).reshape(-1)

    df = pd.DataFrame({"g": gen.astype(int), "y": y})
    df["bin"] = pd.qcut(df["y"], q=10, labels=False, duplicates="drop")
    strat = (df["g"].astype(str) + "_" + df["bin"].astype(str)).to_numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        test_size=args.val_split,
        random_state=args.random_seed,
        stratify=strat,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val, gen_val = X[val_idx], y[val_idx], gen[val_idx]
    sample_weight = 1.0 / 30.0 + y_train

    exp_dir = Path(args.output_dir) / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "xgboost": build_xgboost(args.epochs, args.lr, args.random_seed),
        "lightgbm": build_lgbm(args.epochs, args.lr, args.random_seed),
    }

    histories = []
    best_result = None

    for model_name, model in models.items():
        print(f"training {model_name}...", flush=True)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        save_model(model, exp_dir / f"{model_name}.pkl")

        history = build_iteration_history(
            model, model_name, X_val, y_val, gen_val, args.epochs
        )
        histories.append(history)
        best_row = history.loc[history["val_score"].idxmin()].to_dict()
        print(
            f"{model_name} best | iteration {int(best_row['iteration'])} "
            f"| val_score {best_row['val_score']:.5f} "
            f"| g0 {best_row['err_g0']:.5f} | g1 {best_row['err_g1']:.5f}",
            flush=True,
        )
        if best_result is None or best_row["val_score"] < best_result["val_score"]:
            best_result = best_row

    full_history = pd.concat(histories, ignore_index=True)
    full_history.to_csv(exp_dir / "boosting_history.csv", index=False)
    save_curve(full_history, exp_dir)

    test_blob = torch.load(emb_dir / "test.pt")
    X_test = to_numpy(test_blob["embeddings"])
    for model_name, model in models.items():
        model_history = full_history[full_history["model"] == model_name]
        best_iteration = int(model_history.loc[model_history["val_score"].idxmin(), "iteration"])
        test_preds = model_predict(model, model_name, X_test, best_iteration)
        pd.DataFrame(
            {
                "filename": test_blob["filenames"],
                "FaceOcclusion": np.clip(test_preds, 0.0, 1.0),
                "gender": 0.0,
            },
            columns=["filename", "FaceOcclusion", "gender"],
        ).to_csv(exp_dir / f"submission_{model_name}.csv", index=False)

    with (exp_dir / "result.json").open("w", encoding="utf-8") as file:
        json.dump(best_result, file, indent=2)

    print(
        f"best overall: {best_result['model']} iteration {int(best_result['iteration'])} "
        f"| val_score {best_result['val_score']:.5f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
