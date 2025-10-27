#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Lazy import TensorFlow to allow environment checking before heavy import
try:
    import tensorflow as tf
    from tensorflow.keras import layers, optimizers, losses, callbacks
except Exception as e:
    print("TensorFlow import error. Ensure TensorFlow 2.x is installed.", file=sys.stderr)
    raise

def set_global_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class TaskData:
    x_train: np.ndarray  # (N, 784)
    y_train: np.ndarray  # (N,)
    x_val: np.ndarray    # (N_val, 784)
    y_val: np.ndarray    # (N_val,)
    x_test: np.ndarray   # (N_test, 784)
    y_test: np.ndarray   # (N_test,)


def load_mnist_flat(normalize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32")
    x_test = x_test.reshape(-1, 28 * 28).astype("float32")
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    return (x_train, y_train), (x_test, y_test)


def make_tasks(
    num_tasks: int,
    seed: int,
    val_split: float = 0.1,
) -> List[TaskData]:
    """
    Create `num_tasks` permuted MNIST tasks with deterministic permutations.
    Each task uses a fixed pixel permutation generated from np.random.RandomState(seed + t).
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_flat(normalize=True)
    N, D = x_train.shape
    tasks: List[TaskData] = []

    for t in range(num_tasks):
        rs = np.random.RandomState(seed + t)
        perm = rs.permutation(D)
        x_train_p = x_train[:, perm]
        x_test_p = x_test[:, perm]

      
        idx = np.arange(N)
        rs.shuffle(idx)
        split = int(round((1.0 - val_split) * N))
        tr_idx, val_idx = idx[:split], idx[split:]

        tasks.append(
            TaskData(
                x_train=x_train_p[tr_idx],
                y_train=y_train[tr_idx],
                x_val=x_train_p[val_idx],
                y_val=y_train[val_idx],
                x_test=x_test_p,
                y_test=y_test,
            )
        )
    return tasks



def build_mlp(depth: int, units: int = 256, dropout: float = 0.5) -> tf.keras.Model:
    """
    Build an MLP with `depth` hidden layers of `units` each, optional dropout (0..0.5).
    """
    assert depth >= 1, "Depth must be >= 1"
    inputs = layers.Input(shape=(784,), name="input")
    x = inputs
    for i in range(depth):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(dropout, name=f"dropout_{i+1}")(x)
    outputs = layers.Dense(10, activation="softmax", name="logits")(x)
    model = tf.keras.Model(inputs, outputs, name=f"mlp_d{depth}_u{units}_do{dropout}")
    return model


def one_hot(y_true: tf.Tensor, num_classes: int = 10) -> tf.Tensor:
    return tf.one_hot(tf.cast(y_true, tf.int32), num_classes=num_classes)


def get_loss_fn(loss_name: str):
    """
    Supported loss choices per assignment:
      - 'nll'     : SparseCategoricalCrossentropy
      - 'l1'      : L1 distance to one-hot target
      - 'l2'      : L2 distance to one-hot target
      - 'l1_l2'   : L1 + L2 distance to one-hot target
    """
    lname = loss_name.lower()
    if lname in ("nll", "sce", "crossentropy"):
        return losses.SparseCategoricalCrossentropy()
    elif lname == "l1":
        def l1_loss(y_true, y_pred):
            y_th = one_hot(y_true, 10)
            return tf.reduce_mean(tf.abs(y_th - y_pred))
        return l1_loss
    elif lname == "l2":
        def l2_loss(y_true, y_pred):
            y_th = one_hot(y_true, 10)
            return tf.reduce_mean(tf.square(y_th - y_pred))
        return l2_loss
    elif lname in ("l1_l2", "elastic"):
        def l1l2_loss(y_true, y_pred):
            y_th = one_hot(y_true, 10)
            return tf.reduce_mean(tf.abs(y_th - y_pred)) + tf.reduce_mean(tf.square(y_th - y_pred))
        return l1l2_loss
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def get_optimizer(opt_name: str, lr: float = 1e-3):
    lname = opt_name.lower()
    if lname == "adam":
        return optimizers.Adam(learning_rate=lr)
    elif lname == "sgd":
        return optimizers.SGD(learning_rate=lr, momentum=0.0, nesterov=False)
    elif lname in ("rmsprop", "rms"):
        return optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


@dataclass
class RunConfig:
    depth: int
    units: int
    dropout: float
    optimizer: str
    learning_rate: float
    loss: str
    epochs_first: int
    epochs_rest: int
    batch_size: int
    seed: int


@dataclass
class RunResult:
    config: RunConfig
    ACC: float
    BWT: float
    R: List[List[float]] 
    histories: Dict[str, List[float]] 


def evaluate_on_task(model: tf.keras.Model, task: TaskData, batch_size: int = 256) -> float:
    """Return accuracy on task.test"""
    res = model.evaluate(task.x_test, task.y_test, batch_size=batch_size, verbose=0, return_dict=True)
    acc = float(res.get("accuracy") or res.get("acc") or 0.0)
    return acc


def continual_train(tasks: List[TaskData], cfg: RunConfig, outdir: Path) -> RunResult:
    """
    Train a single MLP sequentially across all tasks.
    After finishing task t (0-based), evaluate on tasks [0..t] and fill row t of R.
    Also record validation accuracy per epoch for plotting.
    """
    T = len(tasks)
    R = np.zeros((T, T), dtype=np.float32)
    histories: Dict[str, List[float]] = {}

    model = build_mlp(depth=cfg.depth, units=cfg.units, dropout=cfg.dropout)
    opt = get_optimizer(cfg.optimizer, lr=cfg.learning_rate)
    loss_fn = get_loss_fn(cfg.loss)
    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

    # Callbacks for early logging
    ckpt_dir = outdir / "checkpoints"
    ensure_dir(ckpt_dir)
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    for t, task in enumerate(tasks):
        epochs = cfg.epochs_first if t == 0 else cfg.epochs_rest
        hist = model.fit(
            task.x_train, task.y_train,
            validation_data=(task.x_val, task.y_val),
            epochs=epochs,
            batch_size=cfg.batch_size,
            verbose=2 if t == 0 else 1,
            callbacks=cbs,
        )
        
        histories[f"task{t+1}"] = list(map(float, hist.history.get("val_accuracy", [])))
		
        for i in range(T):
            if i <= t:
                acc = evaluate_on_task(model, tasks[i])
                R[t, i] = acc
            else:
                R[t, i] = 0.0  
    
    ACC = float(np.mean(R[T - 1, :]))
    diag = np.diag(R)
    BWT = float(np.mean(R[T - 1, :-1] - diag[:-1]))

    return RunResult(
        config=cfg,
        ACC=ACC,
        BWT=BWT,
        R=R.tolist(),
        histories=histories,
    )

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt  


def plot_last_row(R: np.ndarray, savepath: Path, title: str):
    plt.figure(figsize=(8, 4.5))
    T = R.shape[0]
    last = R[T - 1, :]
    plt.plot(np.arange(1, T + 1), last, marker="o")
    plt.xlabel("Task index")
    plt.ylabel("Accuracy after finishing Task T")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_val_histories(histories: Dict[str, List[float]], savepath: Path, title: str):
    plt.figure(figsize=(8, 4.5))
    for key, vals in histories.items():
        if not vals:
            continue
        plt.plot(np.arange(1, len(vals) + 1), vals, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def save_results(outdir: Path, result: RunResult):
    ensure_dir(outdir)
    # JSON
    with open(outdir / "metrics.json", "w") as f:
        json.dump(
            {
                "config": asdict(result.config),
                "ACC": result.ACC,
                "BWT": result.BWT,
                "R": result.R,
                "histories": result.histories,
            },
            f,
            indent=2,
        )

    import csv
    with open(outdir / "R_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t\i"] + [f"Task{i+1}" for i in range(len(result.R))])
        for t, row in enumerate(result.R):
            writer.writerow([f"Task{t+1}"] + [f"{v:.4f}" for v in row])

    # Plots
    R_np = np.array(result.R, dtype=np.float32)
    plot_last_row(R_np, outdir / "final_row_accuracy.png",
                  title=f"Final Accuracy per Task (ACC={result.ACC:.4f}, BWT={result.BWT:.4f})")
    plot_val_histories(result.histories, outdir / "val_accuracy_by_task.png",
                       title="Validation Accuracy vs Epoch (per Task)")


def run_experiments(
    depths: List[int],
    losses_list: List[str],
    optimizers_list: List[str],
    dropout: float,
    units: int,
    epochs_first: int,
    epochs_rest: int,
    batch_size: int,
    seed: int,
    outroot: Path,
    learning_rate: float,
    num_tasks: int,
):
    set_global_seed(seed)
    print(f"[Info] Preparing {num_tasks} Permuted-MNIST tasks with seed={seed} ...")
    tasks = make_tasks(num_tasks=num_tasks, seed=seed, val_split=0.1)

    summary_rows = []
    for depth in depths:
        for loss_name in losses_list:
            for opt_name in optimizers_list:
                cfg = RunConfig(
                    depth=depth,
                    units=units,
                    dropout=dropout,
                    optimizer=opt_name,
                    learning_rate=learning_rate,
                    loss=loss_name,
                    epochs_first=epochs_first,
                    epochs_rest=epochs_rest,
                    batch_size=batch_size,
                    seed=seed,
                )
                print(f"\n[Run] depth={depth} | loss={loss_name} | opt={opt_name} | dropout={dropout}")
                run_dir = outroot / f"d{depth}_{loss_name}_{opt_name}"
                ensure_dir(run_dir)
                result = continual_train(tasks, cfg, outdir=run_dir)
                save_results(run_dir, result)
                print(f"[Result] ACC={result.ACC:.4f} | BWT={result.BWT:.4f} | saved to {run_dir}")
                summary_rows.append({
                    "depth": depth,
                    "loss": loss_name,
                    "optimizer": opt_name,
                    "dropout": dropout,
                    "ACC": result.ACC,
                    "BWT": result.BWT,
                    "outdir": str(run_dir),
                })

   
    with open(outroot / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
		
    import csv
    with open(outroot / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depth", "loss", "optimizer", "dropout", "ACC", "BWT", "outdir"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"\n[Done] All runs completed. Summary stored at: {outroot}")


def parse_args():
    p = argparse.ArgumentParser(
        description="CS 599 - Catastrophic Forgetting on Permuted-MNIST (TensorFlow 2)"
    )
    p.add_argument("--depths", type=str, default="2,3,4",
                   help="Comma-separated hidden depths to test (default: 2,3,4)")
    p.add_argument("--losses", type=str, default="nll,l1,l2,l1_l2",
                   help="Comma-separated losses to test: nll,l1,l2,l1_l2")
    p.add_argument("--optimizers", type=str, default="SGD,Adam,RMSprop",
                   help="Comma-separated optimizers to test")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate <= 0.5 (default: 0.5)")
    p.add_argument("--units", type=int, default=256, help="Hidden units per layer (default: 256)")
    p.add_argument("--epochs_first", type=int, default=50, help="Epochs for Task 1 (default: 50)")
    p.add_argument("--epochs_rest", type=int, default=20, help="Epochs for Tasks 2..T (default: 20)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--seed", type=int, default=1234, help="Global seed (use your unique seed)")
    p.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--num_tasks", type=int, default=10, help="Number of tasks (default: 10)")
    p.add_argument("--outdir", type=str, default="outputs",
                   help="Root output directory (default: ./outputs)")
    return p.parse_args()


def main():
    args = parse_args()
    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    losses_list = [x.strip() for x in args.losses.split(",") if x.strip()]
    optimizers_list = [x.strip() for x in args.optimizers.split(",") if x.strip()]

    if args.dropout < 0.0 or args.dropout > 0.5:
        raise ValueError("Per assignment, use dropout in [0.0, 0.5].")

    outroot = Path(args.outdir) / f"permMNIST_runs_{timestamp()}"
    ensure_dir(outroot)

   
    with open(outroot / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    run_experiments(
        depths=depths,
        losses_list=losses_list,
        optimizers_list=optimizers_list,
        dropout=args.dropout,
        units=args.units,
        epochs_first=args.epochs_first,
        epochs_rest=args.epochs_rest,
        batch_size=args.batch_size,
        seed=args.seed,
        outroot=outroot,
        learning_rate=args.learning_rate,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    main()
