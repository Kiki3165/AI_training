import os
import re
import time
import unicodedata

from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# >>> réseau importé depuis ton fichier existant
from ANN_morelayers import deep_neural_network, predict

# ---------------------- Flask ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
app.config["SECRET_KEY"] = "neuralize-dev"

# ---------------------- Helpers CSV ----------------------
def _normalize(s: str) -> str:
    s = s.strip()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

ALIASES = {
    "surfacemoy": ["surfacemoy", "surface_moy", "surface_moyenne", "surface", "avg_surface", "superficie", "m2"],
    "nbmaisons": ["nbmaisons", "nb_maisons", "nombre_maisons", "houses", "nb"],
    "prixm2moyen": ["prixm2moyen", "prix_m2", "prix_m2_moyen", "prix_metre_carre", "price_per_m2", "prixm2"],
    "prixmoyen": ["prixmoyen", "prix_moyen", "price", "target", "y", "label", "prix"],
}
CANON = {
    "surfacemoy": "SurfaceMoy",
    "nbmaisons": "NbMaisons",
    "prixm2moyen": "Prixm2Moyen",
    "prixmoyen": "PrixMoyen",
}

def auto_read_csv(file_storage) -> pd.DataFrame:
    try:
        file_storage.seek(0)
        return pd.read_csv(file_storage, sep=None, engine="python")
    except Exception:
        file_storage.seek(0)
        return pd.read_csv(file_storage, sep=None, engine="python", encoding="latin-1", errors="ignore")

def map_columns_any_csv(df: pd.DataFrame) -> pd.DataFrame:
    original_cols = list(df.columns)
    norm_map = {_normalize(c): c for c in original_cols}
    found = {}
    for key, syns in ALIASES.items():
        for s in syns:
            if s in norm_map:
                found[key] = norm_map[s]
                break
    missing = [k for k in CANON if k not in found]
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(CANON[k] for k in missing))
    rename_map = {found[k]: CANON[k] for k in found}
    return df.rename(columns=rename_map)

# ---------------------- Plots ----------------------
def generate_plots(df: pd.DataFrame, preds: np.ndarray, history: np.ndarray, ts: int) -> dict:
    out = {}

    def save_current(filename: str):
        full = os.path.join(app.static_folder, filename)
        plt.savefig(full, bbox_inches="tight")
        plt.close()

    # 1) courbe MSE
    plt.figure(figsize=(9, 4))
    plt.plot(history, lw=2)
    plt.title("MSE — Entraînement")
    plt.xlabel("Itérations"); plt.ylabel("MSE"); plt.grid(True, alpha=0.3)
    f1 = f"train_{ts}.png"; save_current(f1); out["train"] = f1

    # 2) hist prix moyen
    plt.figure(figsize=(9, 4))
    plt.hist(df["PrixMoyen"], bins=40)
    plt.title("Distribution du prix moyen (€)")
    plt.xlabel("Prix moyen (€)"); plt.ylabel("Fréquence"); plt.grid(True, alpha=0.3)
    f2 = f"prix_{ts}.png"; save_current(f2); out["prix"] = f2

    # 3) scatter réel
    plt.figure(figsize=(9, 5))
    plt.scatter(df["SurfaceMoy"], df["NbMaisons"], c=df["Prixm2Moyen"], cmap="viridis", edgecolors="k", alpha=0.85)
    plt.colorbar(label="€/m² (réel)")
    plt.title("Surface vs Nb de maisons (réel)")
    plt.xlabel("Surface moyenne (m²)"); plt.ylabel("Nb de maisons"); plt.grid(True, alpha=0.3)
    f3 = f"scatter_{ts}.png"; save_current(f3); out["scatter"] = f3

    # 4) scatter préd
    plt.figure(figsize=(9, 5))
    plt.scatter(df["SurfaceMoy"], df["NbMaisons"], c=preds, cmap="coolwarm", edgecolors="k", alpha=0.85)
    plt.colorbar(label="Prix moyen prédit (€)")
    plt.title("Surface vs Nb de maisons (prédictions)")
    plt.xlabel("Surface moyenne (m²)"); plt.ylabel("Nb de maisons"); plt.grid(True, alpha=0.3)
    f4 = f"scatter_pred_{ts}.png"; save_current(f4); out["scatter_pred"] = f4

    return out

# ---------------------- Routes ----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Nouvelle route AJAX pour l’upload avec barre de progression
@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Aucun fichier CSV sélectionné."}), 400

    # lecture + nettoyage
    df = auto_read_csv(f)
    if df is None or df.empty:
        return jsonify({"error": "CSV vide ou illisible."}), 400

    try:
        df = map_columns_any_csv(df).dropna(how="any")
        for col in ["SurfaceMoy", "NbMaisons", "Prixm2Moyen", "PrixMoyen"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["SurfaceMoy", "NbMaisons", "Prixm2Moyen", "PrixMoyen"])
        if len(df) < 2:
            return jsonify({"error": "Pas assez de lignes valides après nettoyage."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # features / target
    X = df[["SurfaceMoy", "NbMaisons", "Prixm2Moyen"]].T.values.astype(np.float64)
    y = df["PrixMoyen"].values.reshape(1, -1).astype(np.float64)

    # standardisation
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, keepdims=True); sigma = np.where(sigma == 0, 1.0, sigma)
    X_std = (X - mu) / sigma

    y_mean = np.mean(y, axis=1, keepdims=True)
    y_std = np.std(y, axis=1, keepdims=True); y_std = np.where(y_std == 0, 1.0, y_std)
    y_train = (y - y_mean) / y_std

    # entraînement (paramètres raisonnables)
    history, params = deep_neural_network(
        X_std, y_train,
        hidden_layers=(64, 64, 64) if "hidden_layers" in deep_neural_network.__code__.co_varnames else (64, 64),
        learning_rate=1e-4 if "learning_rate" in deep_neural_network.__code__.co_varnames else 1e-4,
        n_iter=300 if "n_iter" in deep_neural_network.__code__.co_varnames else 300,
    )

    # prédiction -> échelle d'origine
    y_pred_std = predict(X_std, params)
    y_pred = y_pred_std * y_std + y_mean
    y_pred_flat = y_pred.flatten()

    # plots
    ts = int(time.time())
    plots = generate_plots(df, y_pred_flat, history.flatten(), ts)

    # Retour JSON : URL de redirection
    redirect_url = url_for("results", **plots)
    return jsonify({"redirect": redirect_url})

@app.route("/results")
def results():
    return render_template(
        "results.html",
        train=request.args.get("train"),
        prix=request.args.get("prix"),
        scatter=request.args.get("scatter"),
        scatter_pred=request.args.get("scatter_pred"),
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
