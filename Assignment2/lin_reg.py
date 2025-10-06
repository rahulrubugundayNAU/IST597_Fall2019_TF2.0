#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math
import argparse
import time
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ----------------------------
# Utilities
# ----------------------------

def name_to_seed(name: str) -> int:
    """
    Convert a first name to a deterministic integer seed by concatenating ASCII codes.
    Example: "Ada" -> "6597" -> 6597
    """
    if not name:
        return 12345
    ascii_concat = "".join(str(ord(c)) for c in name.strip())
    # limit to Python int range; also avoid leading zeros interpreting as octal, etc.
    return int(ascii_concat[:10])  # trim if very long

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Data generation
# ----------------------------

def make_data(n=10_000, x_low=-5.0, x_high=5.0, noise_type="gaussian", noise_level=0.3, seed=123):
    """
    Returns (x, y) with y = 3x + 2 + noise. Noise controlled by type/level.
    noise_level meaning:
        gaussian: std
        laplace : scale (b)
        uniform : half-range (u) -> U[-u, u]
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_low, x_high, size=(n, 1)).astype(np.float32)

    if noise_type == "gaussian":
        eps = rng.normal(0.0, noise_level, size=(n, 1)).astype(np.float32)
    elif noise_type == "laplace":
        eps = rng.laplace(0.0, noise_level, size=(n, 1)).astype(np.float32)
    elif noise_type == "uniform":
        u = noise_level
        eps = rng.uniform(-u, u, size=(n, 1)).astype(np.float32)
    elif noise_type == "none" or noise_level == 0.0:
        eps = np.zeros((n, 1), dtype=np.float32)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    y = 3.0 * x + 2.0 + eps
    return x, y

# ----------------------------
# Model (no Keras)
# ----------------------------

@dataclass
class LinearModel:
    W: tf.Variable
    b: tf.Variable

    @classmethod
    def init(cls, W0=None, b0=None, dtype=tf.float32):
        W_val = 0.0 if W0 is None else float(W0)
        b_val = 0.0 if b0 is None else float(b0)
        return cls(W=tf.Variable(W_val, dtype=dtype), b=tf.Variable(b_val, dtype=dtype))

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self.W * x + self.b

# ----------------------------
# Losses
# ----------------------------

def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    err = y_true - y_pred
    abs_err = tf.abs(err)
    quad = 0.5 * tf.square(err)
    lin  = delta * (abs_err - 0.5 * delta)
    return tf.reduce_mean(tf.where(abs_err <= delta, quad, lin))

def hybrid_loss(y_true, y_pred, lam=0.5):
    # (1-lam)*MSE + lam*MAE
    return (1.0 - lam) * mse_loss(y_true, y_pred) + lam * mae_loss(y_true, y_pred)

def make_loss_fn(loss_name: str, huber_delta: float, hybrid_lambda: float):
    lname = loss_name.lower()
    if lname == "mse":
        return lambda y, yhat: mse_loss(y, yhat)
    if lname == "mae":
        return lambda y, yhat: mae_loss(y, yhat)
    if lname == "huber":
        return lambda y, yhat: huber_loss(y, yhat, delta=huber_delta)
    if lname == "hybrid":
        return lambda y, yhat: hybrid_loss(y, yhat, lam=hybrid_lambda)
    raise ValueError(f"Unknown loss: {loss_name}")

# ----------------------------
# Training
# ----------------------------

def apply_weight_noise(model: LinearModel, std_w: float, std_b: float):
    if std_w > 0.0:
        model.W.assign_add(tf.random.normal(shape=[], stddev=std_w, dtype=model.W.dtype))
    if std_b > 0.0:
        model.b.assign_add(tf.random.normal(shape=[], stddev=std_b, dtype=model.b.dtype))

def train_one_epoch(model, loss_fn, x, y, alpha, batch_size,
                    weight_noise_std_w=0.0, weight_noise_std_b=0.0):
    """Vanilla GD with mini-batches (no TF optimizer/keras), GradientTape updates."""
    n = x.shape[0]
    idx = np.random.permutation(n)
    x = x[idx]
    y = y[idx]
    steps = math.ceil(n / batch_size)

    epoch_loss = 0.0
    for s in range(steps):
        xb = tf.convert_to_tensor(x[s*batch_size:(s+1)*batch_size], dtype=tf.float32)
        yb = tf.convert_to_tensor(y[s*batch_size:(s+1)*batch_size], dtype=tf.float32)

        with tf.GradientTape() as tape:
            yhat = model(xb)
            loss = loss_fn(yb, yhat)

        dW, db = tape.gradient(loss, [model.W, model.b])
        # Gradient descent update (no optimizer)
        model.W.assign_sub(alpha * dW)
        model.b.assign_sub(alpha * db)

        # Optional weight noise after update
        apply_weight_noise(model, weight_noise_std_w, weight_noise_std_b)

        epoch_loss += float(loss.numpy())

    return epoch_loss / steps

def evaluate(model, loss_fn, x, y, batch_size):
    n = x.shape[0]
    steps = math.ceil(n / batch_size)
    total = 0.0
    for s in range(steps):
        xb = tf.convert_to_tensor(x[s*batch_size:(s+1)*batch_size], dtype=tf.float32)
        yb = tf.convert_to_tensor(y[s*batch_size:(s+1)*batch_size], dtype=tf.float32)
        yhat = model(xb)
        total += float(loss_fn(yb, yhat).numpy())
    return total / steps

def maybe_lr_jitter(alpha, lr_noise_std):
    if lr_noise_std <= 0.0:
        return alpha
    jitter = tf.random.normal(shape=[], mean=0.0, stddev=lr_noise_std, dtype=tf.float32).numpy()
    return max(1e-12, alpha * (1.0 + jitter))

def run_training(device_str,
                 loss_name="mse",
                 huber_delta=1.0,
                 hybrid_lambda=0.3,
                 alpha=1e-2,
                 alpha_min=1e-6,
                 epochs=400,
                 batch_size=128,
                 patience=100,
                 lr_noise_std=0.0,
                 weight_noise_std_w=0.0,
                 weight_noise_std_b=0.0,
                 init_W=0.0,
                 init_B=0.0,
                 train_x=None, train_y=None,
                 val_x=None, val_y=None,
                 verbose=True):
    loss_fn = make_loss_fn(loss_name, huber_delta, hybrid_lambda)

    with tf.device(device_str):
        model = LinearModel.init(W0=init_W, b0=init_B)

        best_val = float("inf")
        best_epoch = -1
        best_W, best_B = None, None
        no_improve = 0

        hist = {"train": [], "val": [], "alpha": []}

        start_time = time.time()
        for ep in range(1, epochs + 1):
            current_alpha = maybe_lr_jitter(alpha, lr_noise_std)

            train_loss = train_one_epoch(model, loss_fn, train_x, train_y, current_alpha,
                                         batch_size,
                                         weight_noise_std_w=weight_noise_std_w,
                                         weight_noise_std_b=weight_noise_std_b)
            val_loss = evaluate(model, loss_fn, val_x, val_y, batch_size)

            hist["train"].append(train_loss)
            hist["val"].append(val_loss)
            hist["alpha"].append(current_alpha)

            if verbose and (ep % 25 == 0 or ep == 1):
                print(f"[{device_str}] epoch {ep:4d} | train {train_loss:.6f} | val {val_loss:.6f} | "
                      f"alpha {current_alpha:.3e} | W {model.W.numpy():+.4f} | b {model.b.numpy():+.4f}")

            # Patience scheduling: halve LR on plateau
            if val_loss + 1e-12 < best_val:
                best_val = val_loss
                best_epoch = ep
                best_W, best_B = float(model.W.numpy()), float(model.b.numpy())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    alpha *= 0.5
                    no_improve = 0
                    if verbose:
                        print(f"[{device_str}] patience triggered -> alpha halved to {alpha:.3e}")
                    if alpha < alpha_min:
                        if verbose:
                            print(f"[{device_str}] alpha < alpha_min; early stopping at epoch {ep}")
                        break

        wall_time = time.time() - start_time

    results = {
        "history": hist,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_W": best_W,
        "best_B": best_B,
        "final_W": float(model.W.numpy()),
        "final_B": float(model.b.numpy()),
        "wall_time": wall_time,
        "device": device_str,
        "loss_name": loss_name
    }
    return results

# ----------------------------
# Plotting
# ----------------------------

def plot_fit(x, y, W, B, outdir, title_suffix=""):
    plt.figure(figsize=(6,4))
    # Subsample for scatter
    idx = np.random.choice(len(x), size=min(1200, len(x)), replace=False)
    plt.scatter(x[idx], y[idx], s=6, alpha=0.4, label="data")
    xs = np.linspace(np.min(x), np.max(x), 200).astype(np.float32).reshape(-1,1)
    ys = W * xs + B
    plt.plot(xs, ys, color="crimson", linewidth=2.0, label=f"fit: y={W:.3f}x+{B:.3f}")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Linear fit {title_suffix}".strip())
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"fit{title_suffix.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_losses(train_hist, val_hist, outdir, title_suffix=""):
    plt.figure(figsize=(6,4))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title(f"Loss curves {title_suffix}".strip())
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, f"loss{title_suffix.replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_timing_bar(cpu_ms, gpu_ms, outdir):
    labels = []
    values = []
    colors = []
    if cpu_ms is not None:
        labels.append("CPU")
        values.append(cpu_ms)
        colors.append("#1f77b4")
    if gpu_ms is not None:
        labels.append("GPU")
        values.append(gpu_ms)
        colors.append("#2ca02c")
    if not values:
        return None
    plt.figure(figsize=(4,4))
    plt.bar(labels, values, color=colors)
    plt.ylabel("ms / epoch")
    plt.title("CPU vs GPU timing")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02 * max(values), f"{v:.1f}", ha="center")
    plt.tight_layout()
    path = os.path.join(outdir, "timing_cpu_vs_gpu.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--x_low", type=float, default=-5.0)
    parser.add_argument("--x_high", type=float, default=5.0)
    parser.add_argument("--noise_data", type=str, default="gaussian", choices=["gaussian","laplace","uniform","none"])
    parser.add_argument("--noise_data_level", type=float, default=0.3)
    parser.add_argument("--val_split", type=float, default=0.2)

    # Model/training
    parser.add_argument("--loss", type=str, default="mse", choices=["mse","mae","huber","hybrid"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--hybrid_lambda", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--alpha_min", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=100)

    # Noise outside data
    parser.add_argument("--noise_weights", type=float, default=0.0, help="std for W (and b) noise per step")
    parser.add_argument("--noise_weights_b", type=float, default=None, help="std for b; default equals noise_weights")
    parser.add_argument("--noise_lr", type=float, default=0.0, help="learning-rate jitter std (relative)")
    parser.add_argument("--init_W", type=float, default=0.0)
    parser.add_argument("--init_B", type=float, default=0.0)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_from_name", type=str, default=None, help='e.g., "YourName" -> ASCII-based seed')

    # Devices / timing
    parser.add_argument("--device", type=str, default=None, help='e.g., "/CPU:0" or "/GPU:0"; default: auto')
    parser.add_argument("--time_both_devices", action="store_true", help="Train on CPU then GPU to compare ms/epoch")
    parser.add_argument("--no_plots", action="store_true")

    # IO
    parser.add_argument("--outdir", type=str, default="outputs")

    args = parser.parse_args()

    # Seed
    if args.seed is not None:
        seed_val = args.seed
    elif args.seed_from_name:
        seed_val = name_to_seed(args.seed_from_name)
    else:
        seed_val = 12345
    set_global_seed(seed_val)

    ensure_dir(args.outdir)

    # Data
    x, y = make_data(n=args.n,
                     x_low=args.x_low,
                     x_high=args.x_high,
                     noise_type=args.noise_data,
                     noise_level=args.noise_data_level,
                     seed=seed_val)

    # Train/Val split
    n = x.shape[0]
    val_n = int(args.val_split * n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    train_x, train_y = x[tr_idx], y[tr_idx]
    val_x, val_y = x[val_idx], y[val_idx]

    # Weight noise defaults
    std_w = max(0.0, float(args.noise_weights))
    std_b = std_w if args.noise_weights_b is None else max(0.0, float(args.noise_weights_b))

    # Choose device
    chosen_device = args.device
    if chosen_device is None:
        # prefer GPU if available
        gpus = tf.config.list_physical_devices("GPU")
        chosen_device = "/GPU:0" if gpus else "/CPU:0"

    # Train on chosen device
    res = run_training(
        device_str=chosen_device,
        loss_name=args.loss,
        huber_delta=args.huber_delta,
        hybrid_lambda=args.hybrid_lambda,
        alpha=args.alpha,
        alpha_min=args.alpha_min,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr_noise_std=args.noise_lr,
        weight_noise_std_w=std_w,
        weight_noise_std_b=std_b,
        init_W=args.init_W,
        init_B=args.init_B,
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        verbose=True
    )

    # Report
    ms_per_epoch = 1000.0 * res["wall_time"] / max(1, len(res["history"]["train"]))
    print("\n=== SUMMARY ===")
    print(f"Device: {res['device']}")
    print(f"Loss: {res['loss_name']}")
    print(f"Best val loss: {res['best_val']:.6f} at epoch {res['best_epoch']}")
    print(f"Final params: W={res['final_W']:+.6f}, b={res['final_B']:+.6f}")
    print(f"Best params : W={res['best_W']:+.6f}, b={res['best_B']:+.6f}")
    print(f"Time: {res['wall_time']:.3f} s total -> {ms_per_epoch:.2f} ms/epoch")

    # Plots
    if not args.no_plots:
        title = f"({args.loss}, noise={args.noise_data}:{args.noise_data_level}, init=({args.init_W},{args.init_B}))"
        fit_path = plot_fit(x, y, res["final_W"], res["final_B"], args.outdir, title_suffix=f" {args.loss}")
        loss_path = plot_losses(res["history"]["train"], res["history"]["val"], args.outdir, title_suffix=f" {args.loss}")
        print(f"Saved: {fit_path}\nSaved: {loss_path}")

    # Optional: timing on both devices
    if args.time_both_devices:
        cpu_ms = None
        gpu_ms = None
        # CPU
        cpu = run_training(
            device_str="/CPU:0",
            loss_name=args.loss,
            huber_delta=args.huber_delta,
            hybrid_lambda=args.hybrid_lambda,
            alpha=args.alpha,
            alpha_min=args.alpha_min,
            epochs=min(200, args.epochs),
            batch_size=args.batch_size,
            patience=args.patience,
            lr_noise_std=args.noise_lr,
            weight_noise_std_w=std_w,
            weight_noise_std_b=std_b,
            init_W=args.init_W,
            init_B=args.init_B,
            train_x=train_x, train_y=train_y,
            val_x=val_x, val_y=val_y,
            verbose=False
        )
        cpu_ms = 1000.0 * cpu["wall_time"] / max(1, len(cpu["history"]["train"]))

        # GPU if exists
        if tf.config.list_physical_devices("GPU"):
            gpu = run_training(
                device_str="/GPU:0",
                loss_name=args.loss,
                huber_delta=args.huber_delta,
                hybrid_lambda=args.hybrid_lambda,
                alpha=args.alpha,
                alpha_min=args.alpha_min,
                epochs=min(200, args.epochs),
                batch_size=args.batch_size,
                patience=args.patience,
                lr_noise_std=args.noise_lr,
                weight_noise_std_w=std_w,
                weight_noise_std_b=std_b,
                init_W=args.init_W,
                init_B=args.init_B,
                train_x=train_x, train_y=train_y,
                val_x=val_x, val_y=val_y,
                verbose=False
            )
            gpu_ms = 1000.0 * gpu["wall_time"] / max(1, len(gpu["history"]["train"]))

        if not args.no_plots:
            timing_path = plot_timing_bar(cpu_ms, gpu_ms, args.outdir)
            if timing_path:
                print(f"Saved: {timing_path}")
        if cpu_ms is not None:
            print(f"CPU ms/epoch: {cpu_ms:.2f}")
        if gpu_ms is not None:
            print(f"GPU ms/epoch: {gpu_ms:.2f}")

if __name__ == "__main__":
    main()
