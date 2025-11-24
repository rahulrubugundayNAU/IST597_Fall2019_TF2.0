# CS 599 – DL: Normalization Assignment
# Single CNN class with norm_type: "none", "batch", "layer", "weight"

import tensorflow as tf
import numpy as np
import time

# ----------------- Global config -----------------
seed = 1234
tf.random.set_seed(seed)
np.random.seed(seed)

batch_size = 64
hidden_size = 100
learning_rate = 0.01
output_size = 10  # Fashion-MNIST has 10 classes


# ----------------- Dataset: Fashion-MNIST -----------------
def load_fashion_mnist(batch_size=64):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension: (B, 28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(50000)
        .batch(batch_size)
    )
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds, (x_train, y_train), (x_test, y_test)


# ----------------- Normalization functions (from scratch) -----------------
def batch_norm(x, gamma, beta, eps=1e-5, training=True):
    """
    BatchNorm over (N, H, W) for each channel.
    x: (B, H, W, C)
    gamma, beta: (1, 1, 1, C)
    """
    axes = [0, 1, 2]
    mean = tf.reduce_mean(x, axis=axes, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=axes, keepdims=True)
    x_hat = (x - mean) / tf.sqrt(var + eps)
    return gamma * x_hat + beta


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    LayerNorm over (H, W, C) for each example.
    x: (B, H, W, C)
    gamma, beta: (1, 1, 1, C)
    """
    axes = [1, 2, 3]
    mean = tf.reduce_mean(x, axis=axes, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=axes, keepdims=True)
    x_hat = (x - mean) / tf.sqrt(var + eps)
    return gamma * x_hat + beta


class WeightNormDense(tf.keras.layers.Layer):
    """
    Fully-connected layer with Weight Normalization:
    w = g * v / ||v||
    """

    def __init__(self, in_dim, out_dim, activation=None, name=None):
        super().__init__(name=name)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

    def build(self, input_shape):
        # v has same shape as a standard dense weight matrix
        self.v = self.add_weight(
            name="v",
            shape=[self.in_dim, self.out_dim],
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.g = self.add_weight(
            name="g",
            shape=[self.out_dim],
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        self.b = self.add_weight(
            name="bias",
            shape=[self.out_dim],
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, x):
        # x: (B, in_dim)
        v_norm = tf.norm(self.v, axis=0)  # (out_dim,)
        w = self.v * (self.g / (v_norm + 1e-8))
        y = tf.matmul(x, w) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


# ----------------- CNN model (Option A: norm_type switch) -----------------
class CNN(tf.keras.Model):
    """
    Single CNN class.

    norm_type:
      - "none"  : no normalization
      - "batch" : BatchNorm from scratch
      - "layer" : LayerNorm from scratch
      - "weight": WeightNorm on first FC layer
    """

    def __init__(self, hidden_size, output_size, norm_type="none", device=None, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.size_output = output_size
        self.norm_type = norm_type
        self.device = device

        # Convolutional backbone (Keras conv + maxpool allowed)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, padding="same", use_bias=False
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, padding="same", use_bias=False
        )
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.flatten_layer = tf.keras.layers.Flatten()

        # Gamma/Beta for conv layers if BN or LN
        if self.norm_type in ["batch", "layer"]:
            self.gamma1 = tf.Variable(tf.ones([1, 1, 1, 32]), trainable=True, name="gamma1")
            self.beta1  = tf.Variable(tf.zeros([1, 1, 1, 32]), trainable=True, name="beta1")

            self.gamma2 = tf.Variable(tf.ones([1, 1, 1, 64]), trainable=True, name="gamma2")
            self.beta2  = tf.Variable(tf.zeros([1, 1, 1, 64]), trainable=True, name="beta2")

        # FC layers: WeightNorm on fc1 if norm_type == "weight"
        fc_input_dim = 7 * 7 * 64  # 28x28 -> pool2 -> 7x7

        if self.norm_type == "weight":
            self.fc1 = WeightNormDense(
                fc_input_dim, hidden_size, activation=tf.nn.relu, name="wn_fc1"
            )
        else:
            self.fc1 = tf.keras.layers.Dense(hidden_size, activation=tf.nn.relu)

        self.fc_out = tf.keras.layers.Dense(output_size, activation=None)

        # Our own loss function
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # -------- Legacy functions from your old file (kept but unused) --------
    def flatten(self, X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
        X_padded = tf.pad(X, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
        windows = []
        for y in range(out_h):
            for x in range(out_w):
                window = tf.slice(
                    X_padded,
                    [0, y * stride, x * stride, 0],
                    [-1, window_h, window_w, -1],
                )
                windows.append(window)
        stacked = tf.stack(windows)
        return tf.reshape(stacked, [-1, window_c * window_w * window_h])

    def convolution(self, X, W, b, padding, stride):
        # Legacy, not used in new forward
        n, h, w, c = X.shape
        filter_h, filter_w, filter_c, filter_n = W.shape
        out_h = (h + 2 * padding - filter_h) // stride + 1
        out_w = (w + 2 * padding - filter_w) // stride + 1
        X_flat = self.flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, padding)
        W_flat = tf.reshape(W, [filter_h * filter_w * filter_c, filter_n])
        z = tf.matmul(X_flat, W_flat) + b
        return tf.transpose(tf.reshape(z, [out_h, out_w, n, filter_n]), [2, 0, 1, 3])

    def relu(self, X):
        return tf.maximum(X, tf.zeros_like(X))

    def max_pool(self, X, pool_h, pool_w, padding, stride):
        n, h, w, c = X.shape
        out_h = (h + 2 * padding - pool_h) // stride + 1
        out_w = (w + 2 * padding - pool_w) // stride + 1
        X_flat = self.flatten(X, pool_h, pool_w, c, out_h, out_w, stride, padding)
        pool = tf.reduce_max(tf.reshape(X_flat, [out_h, out_w, n, pool_h * pool_w, c]), axis=3)
        return tf.transpose(pool, [2, 0, 1, 3])

    def affine(self, X, W, b):
        n = tf.shape(X)[0]
        X_flat = tf.reshape(X, [n, -1])
        return tf.matmul(X_flat, W) + b

    def softmax(self, X):
        X_centered = X - tf.reduce_max(X)
        X_exp = tf.exp(X_centered)
        exp_sum = tf.reduce_sum(X_exp, axis=1, keepdims=True)
        return X_exp / exp_sum

    def cross_entropy_error(self, yhat, y):
        return -tf.reduce_mean(tf.math.log(tf.reduce_sum(yhat * y, axis=1)))

    # ----------------- New forward / loss / backward -----------------
    def _apply_norm(self, x, gamma, beta, training=True):
        if self.norm_type == "batch":
            return batch_norm(x, gamma, beta, training=training)
        elif self.norm_type == "layer":
            return layer_norm(x, gamma, beta)
        else:
            return x

    def compute_output(self, X, training=True):
        # X: (B, 28, 28, 1)
        x = self.conv1(X)
        if self.norm_type in ["batch", "layer"]:
            x = self._apply_norm(x, self.gamma1, self.beta1, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        # FIXED: pass x positionally, not as X=x
        x = self.conv2(x)
        if self.norm_type in ["batch", "layer"]:
            x = self._apply_norm(x, self.gamma2, self.beta2, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.flatten_layer(x)

        if self.norm_type == "weight":
            x = self.fc1(x)  # WeightNormDense already has activation
        else:
            x = self.fc1(x)

        logits = self.fc_out(x)
        return logits

    def call(self, X, training=False):
        # Keras uses call() for forward pass; __call__ wraps this automatically
        if self.device is not None:
            dev = "/GPU:0" if self.device == "gpu" else "/CPU:0"
            with tf.device(dev):
                return self.compute_output(X, training=training)
        else:
            return self.compute_output(X, training=training)

    # Optional compatibility wrapper
    def forward(self, X, training=True):
        return self(X, training=training)

    def compute_loss(self, y_pred, y_true):
        """Custom loss method to avoid clashing with keras.Model.loss attribute."""
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        return self.loss_fn(y_true, y_pred)

    def backward(self, X_train, y_train, optimizer):
        with tf.GradientTape() as tape:
            y_pred = self(X_train, training=True)
            current_loss = self.compute_loss(y_pred, y_train)
        grads = tape.gradient(current_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return current_loss


# ----------------- Metrics helpers -----------------
def accuracy_function(logits, true_y):
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    true_y = tf.cast(tf.reshape(true_y, (-1,)), tf.int32)
    correct = tf.equal(preds, true_y)
    return tf.reduce_mean(tf.cast(correct, tf.float32))


# ----------------- Training loop for each norm_type -----------------
def train_model(norm_type, num_epochs=5):
    print(f"\n==== Training with norm_type = {norm_type} ====")
    print("Loading Fashion-MNIST...")
    train_ds, test_ds, (x_train, y_train), (x_test, y_test) = load_fashion_mnist(batch_size)
    print("Finished loading Fashion-MNIST.")

    # Use CPU (device=None) to avoid GPU placement issues
    model = CNN(hidden_size, output_size, norm_type=norm_type, device=None)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for inputs, labels in train_ds:
            loss_value = model.backward(inputs, labels, optimizer)
            epoch_loss += float(loss_value.numpy())
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)

        # Train / test accuracy
        train_logits = model(x_train, training=False)
        train_acc = float(accuracy_function(train_logits, y_train).numpy()) * 100.0

        test_logits = model(x_test, training=False)
        test_acc = float(accuracy_function(test_logits, y_test).numpy()) * 100.0

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%"
        )

    return model


# ----------------- (Optional) Comparison with TF's Norms -----------------
def compare_batchnorm_with_tf(x):
    """
    x: sample activation (B, H, W, C)
    Compares custom BN output with tf.nn.batch_normalization.
    """
    C = x.shape[-1]
    gamma = tf.ones([1, 1, 1, C])
    beta = tf.zeros([1, 1, 1, C])
    eps = 1e-5

    axes = [0, 1, 2]
    mean = tf.reduce_mean(x, axis=axes, keepdims=True)
    var = tf.reduce_mean(tf.square(x - mean), axis=axes, keepdims=True)

    custom = batch_norm(x, gamma, beta, eps, training=True)
    tf_bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

    diff = tf.reduce_max(tf.abs(custom - tf_bn))
    print("Max abs difference (custom BN vs tf BN):", float(diff.numpy()))


def compare_layernorm_with_tf(x):
    """
    x: sample activation (B, H, W, C)
    Compares custom LN output with tf.keras.layers.LayerNormalization.
    """
    C = x.shape[-1]
    gamma = tf.ones([1, 1, 1, C])
    beta = tf.zeros([1, 1, 1, C])
    eps = 1e-5

    custom = layer_norm(x, gamma, beta, eps)

    ln_layer = tf.keras.layers.LayerNormalization(
        axis=[1, 2, 3], epsilon=eps, center=True, scale=True
    )
    _ = ln_layer(x)  # build
    ln_layer.gamma.assign(tf.squeeze(gamma))
    ln_layer.beta.assign(tf.squeeze(beta))

    tf_ln = ln_layer(x)
    diff = tf.reduce_max(tf.abs(custom - tf_ln))
    print("Max abs difference (custom LN vs tf LN):", float(diff.numpy()))


# ----------------- Main -----------------
if __name__ == "__main__":
    start = time.time()

    # Quick run: all four norm types, 1 epoch each (bump epochs later for report)
    for nt in ["none", "batch", "layer", "weight"]:
        _ = train_model(nt, num_epochs=1)

    print("Total time: {:.2f} sec".format(time.time() - start))
