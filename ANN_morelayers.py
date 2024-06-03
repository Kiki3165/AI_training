import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='Train Loss', color='blue')
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='Train MSE', color='green')
    plt.title('Training Mean Squared Error Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return training_history, parametres

def plot_decision_boundary(X, y, parametres):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parametres)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap='Spectral', alpha=0.3)
    plt.scatter(X[0, :], X[1, :], c=y, cmap='Spectral', edgecolors='k')
    plt.xlabel('Surface (m²)')
    plt.ylabel('Nombre de pièces')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.grid(True)
    plt.show()

# Lecture du fichier CSV et préparation des données
file_path = 'dvf2023.csv'
data = pd.read_csv(file_path)

# Supprimer les lignes contenant des NaN
data = data.dropna()

# Sélection des colonnes pour X (caractéristiques) et y (cible)
X = data[['SurfaceMoy', 'NbMaisons', 'Prixm2Moyen']].T.values
y = data['PrixMoyen'].values.reshape(1, -1)

# Normaliser les caractéristiques
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

# Entraînement du modèle avec les nouvelles données nettoyées
training_history, parametres = deep_neural_network(X, y, hidden_layers=(64, 64, 64), learning_rate=1, n_iter=1000)

# Graphique supplémentaire : distribution des prix moyens
plt.figure(figsize=(10, 6))
plt.hist(data['PrixMoyen'], bins=30, color='purple', edgecolor='k', alpha=0.7)
plt.title('Distribution des Prix Moyens')
plt.xlabel('Prix Moyen')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()

# Graphique supplémentaire : scatter plot des deux premières caractéristiques
plt.figure(figsize=(10, 6))
plt.scatter(data['SurfaceMoy'], data['NbMaisons'], c=data['Prixm2Moyen'], cmap='viridis', edgecolors='k', alpha=0.7)
plt.colorbar(label='Prix Moyen')
plt.title('Scatter Plot de Surface vs Nombre de Pièces')
plt.xlabel('Surface (m²)')
plt.ylabel('Nombre de pièces')
plt.grid(True)
plt.show()

# Prédictions des prix moyens
y_pred = predict(X, parametres).flatten()

# Graphique des prédictions des prix moyens vs prix réels
plt.figure(figsize=(10, 6))
plt.scatter(data['SurfaceMoy'], data['NbMaisons'], c=y_pred, cmap='coolwarm', edgecolors='k', alpha=0.7)
plt.colorbar(label='Prix Moyen Prédit')
plt.title('Scatter Plot de Surface vs Nombre de Pièces avec Prédictions de Prix Moyens')
plt.xlabel('Surface (m²)')
plt.ylabel('Nombre de pièces')
plt.grid(True)
plt.show()
