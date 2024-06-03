Projet Deep Neural Network
==========================

Ce projet implémente un réseau de neurones profond (DNN) en utilisant Python et les bibliothèques Numpy, Scikit-learn, tqdm et joblib. Le DNN est entraîné sur un ensemble de données de prévisions immobilières pour prédire les prix moyens des maisons en fonction de diverses caractéristiques.

Fonctionnalités
---------------

- Implémente un DNN avec des couches cachées et des fonctions d'activation ReLU.
- Utilise l'initialisation de Xavier pour initialiser les paramètres du réseau de manière efficace.
- Entraîne le réseau en utilisant la rétropropagation du gradient avec une descente de gradient stochastique.
- Utilise la validation croisée pour surveiller les performances du modèle pendant l'entraînement.
- Fournit une interface pour charger des données à partir de fichiers CSV et afficher les résultats de l'entraînement.

Installation
------------

1. **Clonez ce dépôt** sur votre machine locale.

   ```bash
   git clone https://github.com/votre-utilisateur/deep-neural-network.git
   ```

2. **Assurez-vous d'avoir Python installé** (version 3.x recommandée).

3. **Installez les dépendances** en accédant au répertoire du projet et en exécutant :

   ```bash
   cd deep-neural-network
   pip install -r requirements.txt
   ```

Utilisation
-----------

1. **Placez vos données d'entraînement** dans un fichier CSV.

2. **Modifiez les paramètres d'entraînement et les configurations du réseau** dans le fichier `app.py` si nécessaire.

3. **Lancez l'application** en exécutant :

   ```bash
   python app.py
   ```

4. **Suivez les instructions dans l'interface utilisateur** pour charger les données et entraîner le réseau.

Auteur
------

Ce projet a été développé par Terrasson Kyllian.