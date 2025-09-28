import numpy as np
from sklearn.metrics import mean_squared_error

# --- Activations ---
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(Z.dtype)

# --- Initialisation (He pour ReLU) ---
def he_initialisation(dimensions, seed=1, dtype=np.float64):
    params = {}
    C = len(dimensions) - 1
    rng = np.random.default_rng(seed)
    for c in range(1, C + 1):
        fan_in = dimensions[c - 1]
        fan_out = dimensions[c]
        params[f"W{c}"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_out, fan_in)).astype(dtype)
        params[f"b{c}"] = np.zeros((fan_out, 1), dtype=dtype)
    return params

def initialisation(dimensions, dtype=np.float64):
    return he_initialisation(dimensions, dtype=dtype)

# --- Forward : ReLU cachées, sortie linéaire ---
def forward_propagation(X, params):
    cache = {"A0": X}
    C = len(params) // 2
    for c in range(1, C + 1):
        W = params[f"W{c}"]
        b = params[f"b{c}"]
        A_prev = cache[f"A{c-1}"]
        Z = W @ A_prev + b
        cache[f"Z{c}"] = Z
        cache[f"A{c}"] = relu(Z) if c < C else Z
    return cache

# --- Backprop : MSE ---
def back_propagation(y, params, cache):
    grads = {}
    m = y.shape[1]
    C = len(params) // 2
    A_C = cache[f"A{C}"]
    dA = (2.0 / m) * (A_C - y)

    for c in range(C, 0, -1):
        Z = cache[f"Z{c}"]
        A_prev = cache[f"A{c-1}"]
        W = params[f"W{c}"]

        dZ = dA if c == C else dA * relu_derivative(Z)
        grads[f"dW{c}"] = dZ @ A_prev.T
        grads[f"db{c}"] = np.sum(dZ, axis=1, keepdims=True)
        dA = W.T @ dZ
    return grads

def clip_grads_(grads, max_norm=5.0):
    """Clip L2 des gradients pour éviter l'explosion."""
    total_sq = 0.0
    keys = [k for k in grads if k.startswith("dW")] + [k for k in grads if k.startswith("db")]
    for k in keys:
        g = grads[k]
        total_sq += np.sum(g * g)
    norm = np.sqrt(total_sq)
    if not np.isfinite(norm) or norm == 0.0:
        return grads
    scale = max_norm / max(norm, max_norm)
    if scale < 1.0:
        for k in keys:
            grads[k] *= scale
    return grads

def update(grads, params, lr):
    C = len(params) // 2
    for c in range(1, C + 1):
        params[f"W{c}"] -= lr * grads[f"dW{c}"]
        params[f"b{c}"] -= lr * grads[f"db{c}"]
    return params

def predict(X, params):
    cache = forward_propagation(X, params)
    C = len(params) // 2
    return cache[f"A{C}"]

# --- Entraînement ---
def deep_neural_network(X, y, hidden_layers=(124, 124, 124), learning_rate=1e-4, n_iter=500, n_jobs=None):
    """
    X: (n_features, m) float64
    y: (n_outputs, m) float64 (pour régression univariée: (1, m)), idéalement standardisé
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    dims = [X.shape[0], *hidden_layers, y.shape[0]]
    params = initialisation(dims, dtype=np.float64)

    history = np.zeros((int(n_iter), 2), dtype=np.float64)
    last_mse = np.inf
    patience = 25
    worse = 0

    for i in range(n_iter):
        cache = forward_propagation(X, params)
        y_pred = cache[f"A{len(dims) - 1}"]

        mse = mean_squared_error(y.flatten(), y_pred.flatten())
        if not np.isfinite(mse):
            history = history[: i]
            break

        history[i, 0] = mse
        history[i, 1] = mse

        grads = back_propagation(y, params, cache)
        grads = clip_grads_(grads, max_norm=5.0)
        params = update(grads, params, learning_rate)

        if mse > last_mse * 1.1:
            worse += 1
        else:
            worse = 0
        last_mse = mse
        if worse >= patience:
            history = history[: i + 1]
            break

    return history, params
