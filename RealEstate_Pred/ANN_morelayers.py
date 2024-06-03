import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def tanh(Z):
    return np.tanh(Z)

def tanh_derivative(Z):
    return 1 - np.power(np.tanh(Z), 2)

def initialisation(dimensions):
    parametres = {}
    C = len(dimensions)
    np.random.seed(1)
    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1]) * 0.01
        parametres['b' + str(c)] = np.zeros((dimensions[c], 1))
    return parametres

def forward_propagation(X, parametres):
    activations = {'A0': X}
    C = len(parametres) // 2
    for c in range(1, C + 1):
        Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
        activations['A' + str(c)] = tanh(Z)
    return activations

def back_propagation(y, parametres, activations):
    m = y.shape[1]
    C = len(parametres) // 2
    dZ = activations['A' + str(C)] - y
    gradients = {}
    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parametres['W' + str(c)].T, dZ) * tanh_derivative(activations['A' + str(c - 1)])
    return gradients

def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2
    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]
    return parametres

def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2
    Af = activations['A' + str(C)]
    return Af

def deep_neural_network(X, y, hidden_layers=(16, 16, 16), learning_rate=1, n_iter=3000):
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)
    training_history = np.zeros((int(n_iter), 2))
    C = len(parametres) // 2
    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parametres)
        gradients = back_propagation(y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]
        training_history[i, 0] = mean_squared_error(y.flatten(), Af.flatten())
        y_pred = predict(X, parametres)
        training_history[i, 1] = mean_squared_error(y.flatten(), y_pred.flatten())
    return training_history, parametres
