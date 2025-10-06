#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def name_to_seed(name: str) -> int:
    """Convert a name to a deterministic integer seed."""
    return int("".join(str(ord(c)) for c in name)[:10]) if name else 12345

def set_global_seed(seed: int):
    """Ensure deterministic behavior as much as possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape([-1, 784])
    x_test  = x_test.reshape([-1, 784])
    return (x_train, y_train), (x_test, y_test)

def train_val_split(x, y, val_split=0.1, seed=123):
    """Split into train and validation sets."""
    n = x.shape[0]
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    n_val = int(val_split * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return (x[tr_idx], y[tr_idx]), (x[val_idx], y[val_idx])

# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def plot_images(x, y_true, y_pred=None, outdir="outputs", n=10, title="sample"):
    plt.figure(figsize=(10,2))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(x[i].reshape(28,28), cmap="gray")
        lbl = f"T:{y_true[i]}"
        if y_pred is not None:
            lbl += f"/P:{y_pred[i]}"
        plt.title(lbl, fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    ensure_dir(outdir)
    path = os.path.join(outdir, f"images_{title}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_weights(W, outdir="outputs", name="adam"):
    """Visualize learned class weights as 28x28 grayscale images."""
    ensure_dir(outdir)
    plt.figure(figsize=(10,4))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(W[i].reshape(28,28), cmap="gray")
        plt.axis("off")
        plt.title(f"class {i}")
    plt.tight_layout()
    path = os.path.join(outdir, f"weights_{name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

# ----------------------------------------------------------------------
# Model (no Keras)
# ----------------------------------------------------------------------

class LogisticRegressionTF:
    def __init__(self, num_features=784, num_classes=10, l2_lambda=0.0):
        self.W = tf.Variable(tf.zeros([num_classes, num_features]), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)
        self.l2_lambda = l2_lambda

    def predict_logits(self, x):
        return tf.matmul(x, tf.transpose(self.W)) + self.b

    def predict_proba(self, x):
        return tf.nn.softmax(self.predict_logits(x))

    def loss(self, x, y):
        """Compute loss and gradients."""
        y = tf.cast(y, tf.int32)  # ✅ FIX: ensure labels are int32
        with tf.GradientTape() as tape:
            logits = self.predict_logits(x)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
            )
            if self.l2_lambda > 0:
                loss += self.l2_lambda * tf.nn.l2_loss(self.W)
        grads = tape.gradient(loss, [self.W, self.b])
        return loss, grads

# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------

def run_training(model, optimizer, train_data, val_data, epochs=30, batch_size=128, device="/CPU:0"):
    (x_train, y_train) = train_data
    (x_val, y_val) = val_data
    n = x_train.shape[0]
    steps = n // batch_size
    hist = {"train_acc": [], "val_acc": []}
    start = time.time()

    with tf.device(device):
        for ep in range(1, epochs+1):
            idx = np.random.permutation(n)
            x_train, y_train = x_train[idx], y_train[idx]
            for i in range(steps):
                xb = tf.convert_to_tensor(x_train[i*batch_size:(i+1)*batch_size])
                yb = tf.convert_to_tensor(y_train[i*batch_size:(i+1)*batch_size])
                loss, grads = model.loss(xb, yb)
                optimizer.apply_gradients(zip(grads, [model.W, model.b]))

            # Metrics
            train_preds = tf.argmax(model.predict_proba(x_train), axis=1)
            val_preds   = tf.argmax(model.predict_proba(x_val), axis=1)
            train_acc = np.mean((train_preds.numpy()==y_train).astype(np.float32))
            val_acc   = np.mean((val_preds.numpy()==y_val).astype(np.float32))
            hist["train_acc"].append(train_acc)
            hist["val_acc"].append(val_acc)
            if ep%5==0 or ep==1:
                print(f"[{device}] epoch {ep:03d}: train={train_acc:.3f}, val={val_acc:.3f}")

    hist["time_per_epoch"] = (time.time()-start)/epochs
    return hist

# ----------------------------------------------------------------------
# Baseline Models
# ----------------------------------------------------------------------

def baseline_random_forest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1)
    clf.fit(x_train, y_train)
    return accuracy_score(y_test, clf.predict(x_test))

def baseline_svm(x_train, y_train, x_test, y_test, n=5000):
    idx = np.random.choice(len(x_train), size=n, replace=False)
    clf = svm.LinearSVC(max_iter=5000)
    clf.fit(x_train[idx], y_train[idx])
    return accuracy_score(y_test, clf.predict(x_test))

# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd","adam","rmsprop"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed_from_name", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compare_cpu_gpu", action="store_true")
    args = parser.parse_args()

    seed = name_to_seed(args.seed_from_name) if args.seed_from_name else 12345
    set_global_seed(seed)
    ensure_dir(args.outdir)

    # Load data
    (x_train_full, y_train_full), (x_test, y_test) = load_fashion_mnist()
    (train_x, train_y), (val_x, val_y) = train_val_split(x_train_full, y_train_full,
                                                         val_split=args.val_split, seed=seed)

    # Model & Optimizer
    model = LogisticRegressionTF(l2_lambda=args.l2)
    opt = {"sgd": tf.optimizers.SGD(1e-2),
           "adam": tf.optimizers.Adam(1e-3),
           "rmsprop": tf.optimizers.RMSprop(1e-3)}[args.optimizer]

    device = args.device or ("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0")
    print(f"Training on {device} with {args.optimizer.upper()} optimizer")

    hist = run_training(model, opt, (train_x, train_y), (val_x, val_y),
                        epochs=args.epochs, batch_size=args.batch_size, device=device)

    # Evaluate
    test_preds = tf.argmax(model.predict_proba(x_test), axis=1).numpy()
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Test accuracy ({args.optimizer.upper()}): {test_acc*100:.2f}%")
    print(f"Mean time per epoch: {hist['time_per_epoch']:.3f}s")

    # Visualizations
    plot_images(x_test, y_test, test_preds, outdir=args.outdir, title=args.optimizer)
    plot_weights(model.W.numpy(), outdir=args.outdir, name=args.optimizer)

    # CPU vs GPU timing
    if args.compare_cpu_gpu and tf.config.list_physical_devices("GPU"):
        cpu_model = LogisticRegressionTF(l2_lambda=args.l2)
        cpu_opt = tf.optimizers.Adam(1e-3)
        cpu_hist = run_training(cpu_model, cpu_opt, (train_x, train_y), (val_x, val_y),
                                epochs=5, batch_size=args.batch_size, device="/CPU:0")
        gpu_hist = run_training(model, opt, (train_x, train_y), (val_x, val_y),
                                epochs=5, batch_size=args.batch_size, device="/GPU:0")
        cpu_ms = 1000*cpu_hist["time_per_epoch"]
        gpu_ms = 1000*gpu_hist["time_per_epoch"]
        plt.bar(["CPU","GPU"], [cpu_ms, gpu_ms], color=["#1f77b4","#2ca02c"])
        plt.ylabel("ms / epoch")
        plt.title("CPU vs GPU Timing")
        path = os.path.join(args.outdir,"timing_cpu_gpu_logreg.png")
        plt.savefig(path,dpi=150); plt.close()
        print(f"Saved: {path}")

    # Baselines
    print("\n--- Baseline Comparisons ---")
    try:
        rf_acc = baseline_random_forest(train_x[:10000], train_y[:10000], x_test, y_test)
        svm_acc = baseline_svm(train_x[:5000], train_y[:5000], x_test, y_test)
        print(f"RandomForest: {rf_acc*100:.2f}% | SVM: {svm_acc*100:.2f}%")
    except Exception as e:
        print("Baseline models skipped due to error:", e)

if __name__ == "__main__":
    main()
