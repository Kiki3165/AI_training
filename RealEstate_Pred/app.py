from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ANN_morelayers import deep_neural_network, predict, initialisation, forward_propagation, back_propagation, update

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comments.db'
db = SQLAlchemy(app)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    content = db.Column(db.Text)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            data = pd.read_csv(file)
            data = data.dropna()

            X = data[['SurfaceMoy', 'NbMaisons', 'Prixm2Moyen']].T.values
            y = data['PrixMoyen'].values.reshape(1, -1)
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

            hidden_layers = (64, 64, 64)
            learning_rate = 1
            n_iter = 500

            training_history, parametres = deep_neural_network(X, y, hidden_layers, learning_rate, n_iter)

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
            plt.savefig('static/training_history.png')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.hist(data['PrixMoyen'], bins=30, color='purple', edgecolor='k', alpha=0.7)
            plt.title('Distribution des Prix Moyens')
            plt.xlabel('Prix Moyen')
            plt.ylabel('Fréquence')
            plt.grid(True)
            plt.savefig('static/distribution_prix.png')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.scatter(data['SurfaceMoy'], data['NbMaisons'], c=data['Prixm2Moyen'], cmap='viridis', edgecolors='k', alpha=0.7)
            plt.colorbar(label='Prix au m² Moyen')
            plt.title('Scatter Plot de Surface Moyenne vs Nombre de Maisons')
            plt.xlabel('Surface Moyenne (m²)')
            plt.ylabel('Nombre de Maisons')
            plt.grid(True)
            plt.savefig('static/scatter_plot.png')
            plt.close()

            y_pred = predict(X, parametres).flatten()

            plt.figure(figsize=(10, 6))
            plt.scatter(data['SurfaceMoy'], data['NbMaisons'], c=y_pred, cmap='coolwarm', edgecolors='k', alpha=0.7)
            plt.colorbar(label='Prix Moyen Prédit')
            plt.title('Scatter Plot de Surface Moyenne vs Nombre de Maisons avec Prédictions de Prix Moyens')
            plt.xlabel('Surface Moyenne (m²)')
            plt.ylabel('Nombre de Maisons')
            plt.grid(True)
            plt.savefig('static/scatter_plot_predictions.png')
            plt.close()

            return redirect(url_for('results'))
    comments = Comment.query.all()
    return render_template("index.html", comments=comments)

@app.route("/results")
def results():
    return render_template("results.html")

@app.route("/comment", methods=["POST"])
def comment():
    username = request.form["username"]
    content = request.form["content"]
    comment = Comment(username=username, content=content)
    db.session.add(comment)
    db.session.commit()
    return redirect(url_for("comment_page"))

@app.route("/comments")
def comments():
    comments = Comment.query.all()
    return render_template("comments.html", comments=comments)

@app.route("/login", methods=["POST"])
def login():
    # Traitement de la connexion
    return redirect(url_for("index"))

@app.route("/register", methods=["POST"])
def register():
    # Traitement de l'inscription
    return redirect(url_for("index"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
