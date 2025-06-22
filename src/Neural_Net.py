import struct
import numpy as np

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((num_images, rows, cols))

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

# ─── Activations ──────────────────────────────────────────────────────────────

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_derivative(Z):
    return (Z > 0).astype(np.float32)

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# ─── One‐hot encoding ─────────────────────────────────────────────────────────

def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    Y = np.zeros((num_classes, m))
    Y[y, np.arange(m)] = 1
    return Y

# ─── Parameter initialization ─────────────────────────────────────────────────

def initialize_paras(n_x, n_h1, n_h2, n_y):
    np.random.seed(1)
    params = {}
    # He initialization for ReLU
    params["W1"] = np.random.randn(n_h1, n_x) * np.sqrt(2. / n_x)
    params["b1"] = np.zeros((n_h1, 1))
    params["W2"] = np.random.randn(n_h2, n_h1) * np.sqrt(2. / n_h1)
    params["b2"] = np.zeros((n_h2, 1))
    params["W3"] = np.random.randn(n_y, n_h2) * np.sqrt(2. / n_h2)
    params["b3"] = np.zeros((n_y, 1))
    return params

# ─── Forward / Cost / Backward ────────────────────────────────────────────────

def forward_prop(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = ReLU(Z2)
    Z3 = W3 @ A2 + b3
    A3 = softmax(Z3)

    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

def cost_function(A3, Y):
    # cross‐entropy
    m = Y.shape[1]
    eps = 1e-15
    A3 = np.clip(A3, eps, 1-eps)
    loss = -np.sum(Y * np.log(A3)) / m
    return loss

def backward_prop(parameters, cache, X, Y):
    m = X.shape[1]
    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]
    Z1, A1, Z2, A2, Z3, A3 = cache

    dZ3 = A3 - Y
    dW3 = dZ3 @ A2.T / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m

    dA2 = W3.T @ dZ3
    dZ2 = dA2 * ReLU_derivative(Z2)
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dA1 = W2.T @ dZ2
    dZ1 = dA1 * ReLU_derivative(Z1)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2,
            "dW3": dW3, "db3": db3}

def update_parameters(parameters, grads, learning_rate):
    for l in (1, 2, 3):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

# ─── Mini‐batch generator ────────────────────────────────────────────────────

def get_mini_batches(X, Y, batch_size):
    m = X.shape[1]
    perm = np.random.permutation(m)
    X, Y = X[:, perm], Y[:, perm]
    for i in range(0, m, batch_size):
        yield X[:, i:i+batch_size], Y[:, i:i+batch_size]

# ─── Model training ────────────────────────────────────────────────────────────

def nn_model(X, Y, epochs=10, batch_size=64, lr=0.1, print_cost=False):
    n_x, m = X.shape
    n_y, _ = Y.shape
    # fixed hidden dims:
    parameters = initialize_paras(n_x, 128, 64, n_y)

    for epoch in range(1, epochs+1):
        epoch_cost = 0
        n_batches = 0

        for X_batch, Y_batch in get_mini_batches(X, Y, batch_size):
            A3, cache = forward_prop(X_batch, parameters)
            cost = cost_function(A3, Y_batch)
            grads = backward_prop(parameters, cache, X_batch, Y_batch)
            parameters = update_parameters(parameters, grads, lr)
            epoch_cost += cost
            n_batches += 1

        if print_cost:
            print(f"Epoch {epoch}/{epochs} — avg cost: {epoch_cost/n_batches:.4f}")

    return parameters

# ─── Prediction ────────────────────────────────────────────────────────────────

def predict(parameters, X):
    A3, _ = forward_prop(X, parameters)
    return np.argmax(A3, axis=0)

# ─── Putting it all together ──────────────────────────────────────────────────

# 1. Load
train_images = load_images('data/train-images.idx3-ubyte')
train_labels = load_labels('data/train-labels.idx1-ubyte')

# 2. Preprocess
X = train_images.reshape(-1, 28*28).T.astype(np.float32) / 255.0
Y = one_hot_encode(train_labels)

# 3. Train
params = nn_model(X, Y,
                  epochs=10,        # 10 passes over the data
                  batch_size=128,   # 128 images per gradient step
                  lr=0.1,
                  print_cost=True)

# 4. Evaluate on train set
preds = predict(params, X)
print("Train accuracy:", np.mean(preds == train_labels) * 100, "%")

test_images = load_images('data/t10k-images.idx3-ubyte')
test_labels = load_labels('data/t10k-labels.idx1-ubyte')

X_test = test_images.reshape(-1, 28*28).T.astype(np.float32) / 255.0
Y_test = one_hot_encode(test_labels)

preds = predict(params,X_test)
print("Test accuracy:", np.mean(preds == test_labels) * 100, "%")

