import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from sklearn.svm import SVC
from matplotlib.animation import FuncAnimation

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T  # Transposer les données X
y = y.reshape((1, y.shape[0]))

def initialisation(n0, n1, n2):

    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)
    
    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }
    
    return parametres

def forward_propagation(X, parametres):
    
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']
    
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    
    activations = {
        'A1': A1,
        'A2': A2
    }
    
    return activations

def back_propagation(X, y, activations, parametres):
    
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']
    
    m  = y.shape[1]
    
    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
    }
    
    return gradients

def update(gradients, parametres, learning_rate):
    
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres

def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    A2 = activations['A2']
    return A2 >= 0.5

def neural_network(X_train, y_train, n1, learning_rate=1, n_iter=1000):

    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    parametres = initialisation(n0, n1, n2)
    
    train_loss = []
    train_acc = []
    
    for i in range(n_iter):
        
        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)
        
        if i % 10 == 0:
            train_loss.append(log_loss(y_train.flatten(), activations['A2'].flatten()))
            y_pred = predict(X_train, parametres)
            current_accuracy = accuracy_score(y_train.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)
            
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train acc')
    plt.legend()
    plt.show()
    
    return parametres

# Appel de la fonction neural_network à l'extérieur
parametres = neural_network(X, y, n1=128, n_iter=1000, learning_rate=0.1)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, log_loss
from matplotlib.animation import FuncAnimation

# Fonctions de votre code pour initialisation, propagation, rétropropagation, mise à jour, prédiction et réseau neuronal
# ...

# Appel de la fonction neural_network à l'extérieur
parametres = neural_network(X, y, n1=64, n_iter=1000, learning_rate=2)

def update_plot(frame):
    global X, y, ax, parametres
    # Mettre à jour les données d'entrée
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=frame)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    
    ax.clear()
    xx, yy = np.meshgrid(np.linspace(X[0, :].min(), X[0, :].max(), 100),
                         np.linspace(X[1, :].min(), X[1, :].max(), 100))
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, parametres)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.8)  # Utiliser cmap='coolwarm' pour un style météo
    ax.scatter(X[0, :], X[1, :], c=y, cmap='coolwarm', edgecolors='black')
    plt.title('Frontière de décision en temps réel', fontsize=14)
    plt.xlabel('X1', fontsize=12)
    plt.ylabel('X2', fontsize=12)
    plt.axis('equal')
    
    # Calculer la précision en temps réel
    y_pred = predict(X, parametres)
    current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
    plt.text(0.95, 0.95, 'Accuracy: {:.2f}%'.format(current_accuracy * 100),
             horizontalalignment='right', verticalalignment='top',
             transform=ax.transAxes, fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

# Générer les données
X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

# Créer la figure et l'axe
fig, ax = plt.subplots(figsize=(10, 6))

# Animer le plot
ani = FuncAnimation(fig, update_plot, interval=500, cache_frame_data=False, frames=120)

# Afficher le plot
plt.show()