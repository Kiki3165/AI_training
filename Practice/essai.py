import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

# Lecture du fichier CSV et préparation des données
file_path = 'dvf2023.csv'
data = pd.read_csv(file_path)

# Supprimer les lignes contenant des NaN
data = data.dropna()

# Sélection des colonnes pour X (caractéristiques) et y (cible)
X = data.drop(columns=['Unnamed: 0', 'INSEE_COM', 'Annee', 'PrixMoyen']).values.T
y = data['PrixMoyen'].values.reshape(1, -1)

# Vérification des valeurs NaN ou infinies
print(np.isnan(X).any(), np.isinf(X).any())
print(np.isnan(y).any(), np.isinf(y).any())