
 # Programme probabiliste Française des jeux
 # 01/02/2025 - 12:11

import argparse
import random
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import f_oneway, binom, uniform, norm, poisson, ttest_ind, chi2_contingency
from collections import Counter
import gzip
import base64
import io
import json
import logging
from typing import List, Dict, Any, Union

# Optionnel : Importation d'optuna pour l'optimisation bayésienne
try:
    import optuna
except ImportError:
    optuna = None

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================
# Fonctions de chargement et de vérification
# ============================

def charger_historique_loto(historique_loto_compressed: str) -> List[Dict[str, Any]]:
    # ... (votre code inchangé)

def verifier_donnees(data: List[Dict[str, Any]]) -> None:
    """
    Vérifie que chaque entrée de l'historique est un dictionnaire avec les clés "main" et "chance",
    que "main" contient 5 nombres uniques entre 1 et 49, et que "chance" est entre 1 et 10.
    """
    if not isinstance(data, list):
        raise TypeError("Les données doivent être une liste.")

    for i, entry in enumerate(data):  # Ajout de l'indice pour un meilleur message d'erreur
        if not isinstance(entry, dict):
            raise TypeError(f"L'entrée {i} doit être un dictionnaire.")

        if "main" not in entry or "chance" not in entry:
            raise ValueError(f"L'entrée {i} doit contenir les clés 'main' et 'chance'.")

        main = entry["main"]
        chance = entry["chance"]

        if not isinstance(main, list):
            raise TypeError(f"La clé 'main' de l'entrée {i} doit être une liste.")

        if len(main) != 5:
            raise ValueError(f"La clé 'main' de l'entrée {i} doit contenir exactement 5 nombres.")

        if not all(isinstance(x, int) for x in main):
            raise TypeError(f"Les nombres de la clé 'main' de l'entrée {i} doivent être des entiers.")

        if not all(1 <= x <= 49 for x in main):
            raise ValueError(f"Les nombres de la clé 'main' de l'entrée {i} doivent être compris entre 1 et 49.")

        if len(set(main)) != 5:
            raise ValueError(f"Les nombres de la clé 'main' de l'entrée {i} doivent être uniques.")

        if not isinstance(chance, int):
            raise TypeError(f"La clé 'chance' de l'entrée {i} doit être un entier.")

        if not 1 <= chance <= 10:
            raise ValueError(f"La clé 'chance' de l'entrée {i} doit être comprise entre 1 et 10.")


# Optionnel : Importation d'optuna pour l'optimisation bayésienne
try:
    import optuna
except ImportError:
    optuna = None

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================
# Fonctions de chargement et de vérification
# ============================

def charger_historique_loto(historique_loto_compressed: str) -> List[Dict[str, Any]]:
    """
    Décode et décompresse l'historique des tirages du loto intégré en Base64 et Gzip.
    Chaque tirage est attendu sous la forme d'un dictionnaire avec :
       - "main": liste de 5 nombres (boules principales, de 1 à 49)
       - "chance": un nombre (numéro chance, de 1 à 10)
    """
    try:
        compressed_data = base64.b64decode(historique_loto_compressed)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed_data)) as f:
            data = json.load(f)
        # On suppose que les données sont déjà dans le format voulu.
        logging.info("Chargement et décompression des données réussi.")
        return data
    except (base64.binascii.Error, gzip.BadGzipFile, json.JSONDecodeError, ValueError) as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        return []


def verifier_donnees(data: List[Dict[str, Any]]) -> None:
    """
    Vérifie que chaque entrée de l'historique possède les clés "main" et "chance".
    (Ici, aucune validation poussée n'est effectuée puisque l'intégration réelle respecte ce format.)
    """
    if not isinstance(data, list):
        raise ValueError("Les données doivent être une liste.")
    for entry in data:
        if not isinstance(entry, dict) or "main" not in entry or "chance" not in entry:
            raise ValueError("Chaque entrée doit être un dictionnaire avec les clés 'main' et 'chance'.")


# ============================
# Fonctions d'analyse statistique
# ============================
def calculer_mesures_statistiques(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcule des mesures de tendance centrale et de dispersion séparément pour :
       - Les boules principales (on fusionne toutes les listes)
       - Le numéro chance (liste de nombres)
    """
    verifier_donnees(data)
    # Pour les boules principales
    main_numbers = [num for entry in data for num in entry['main']]
    stats_main = {
        "moyenne": np.mean(main_numbers),
        "mediane": np.median(main_numbers),
        "mode": Counter(main_numbers).most_common(1)[0][0],
        "ecart_type": np.std(main_numbers),
        "variance": np.var(main_numbers),
        "iqr": np.percentile(main_numbers, 75) - np.percentile(main_numbers, 25)
    }
    # Pour le numéro chance
    chance_numbers = [entry['chance'] for entry in data]
    stats_chance = {
        "moyenne": np.mean(chance_numbers),
        "mediane": np.median(chance_numbers),
        "mode": Counter(chance_numbers).most_common(1)[0][0],
        "ecart_type": np.std(chance_numbers),
        "variance": np.var(chance_numbers),
        "iqr": np.percentile(chance_numbers, 75) - np.percentile(chance_numbers, 25)
    }
    stats = {"main": stats_main, "chance": stats_chance}
    logging.info(f"Mesures statistiques calculées : {stats}")
    return stats


def probabilites(data: List[Dict[str, Any]]) -> Dict[str, Dict[Any, float]]:
    """
    Calcule les probabilités d'apparition des numéros séparément pour :
       - Les boules principales (de 1 à 49)
       - Le numéro chance (de 1 à 10)
    """
    verifier_donnees(data)
    logging.info("Calcul des probabilités d'apparition des numéros...")
    main_counts = Counter()
    chance_counts = Counter()
    for entry in data:
        main_counts.update(entry['main'])
        chance_counts.update([entry['chance']])
    total_main = sum(main_counts.values())
    total_chance = sum(chance_counts.values())
    probs_main = {num: count / total_main for num, count in main_counts.items()}
    probs_chance = {num: count / total_chance for num, count in chance_counts.items()}
    probs = {"main": probs_main, "chance": probs_chance}
    logging.info(f"Probabilités des numéros : {probs}")
    return probs


def liens(data: List[Dict[str, Any]]) -> None:
    """
    Recherche des liens entre les numéros des boules principales (uniquement).
    """
    verifier_donnees(data)
    logging.info("Recherche des liens entre les numéros des boules principales...")
    # Créer une matrice de co-occurrence pour les boules principales
    all_main = [entry['main'] for entry in data]
    num_max = 49  # Pour les boules principales
    co_occurrence = np.zeros((num_max + 1, num_max + 1))
    for main in all_main:
        for i in main:
            for j in main:
                if i != j:
                    co_occurrence[i, j] += 1
    # Test du chi² pour chaque paire
    for i in range(1, num_max + 1):
        for j in range(i + 1, num_max + 1):
            if co_occurrence[i, j] > 0:
                contingency_table = np.array([
                    [co_occurrence[i, j], sum(1 for main in all_main if i in main) - co_occurrence[i, j]],
                    [sum(1 for main in all_main if j in main) - co_occurrence[i, j],
                     len(data) - sum(1 for main in all_main if i in main) - sum(1 for main in all_main if j in main) + co_occurrence[i, j]]
                ])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                if p < 0.05:
                    logging.info(f"Lien significatif entre les numéros {i} et {j} (chi2 = {chi2:.2f}, p = {p:.3f})")


def machine(data: List[Dict[str, Any]]) -> None:
    """
    Simule un tirage en s'appuyant sur les probabilités empiriques :
       - Pour les boules principales : 5 numéros tirés selon leur probabilité d'apparition
       - Pour le numéro chance : 1 numéro tiré selon sa probabilité d'apparition
    """
    verifier_donnees(data)
    logging.info("Simulation de tirages...")
    probs = probabilites(data)
    main_numbers = list(probs["main"].keys())
    main_weights = list(probs["main"].values())
    chance_numbers = list(probs["chance"].keys())
    chance_weights = list(probs["chance"].values())
    tirage_main = random.choices(main_numbers, weights=main_weights, k=5)
    tirage_chance = random.choices(chance_numbers, weights=chance_weights, k=1)[0]
    logging.info(f"Tirage simulé : main = {sorted(tirage_main)}, chance = {tirage_chance}")


def series(data: List[Dict[str, Any]]) -> None:
    """
    Affiche l'évolution de la fréquence d'apparition des numéros des boules principales au fil des tirages.
    """
    verifier_donnees(data)
    logging.info("Analyse des séries temporelles pour les boules principales...")
    num_counts = {num: [] for num in range(1, 50)}
    for entry in data:
        for num in range(1, 50):
            num_counts[num].append(entry['main'].count(num))
    plt.figure(figsize=(12, 6))
    for num in num_counts:
        plt.plot(num_counts[num], label=f"Numéro {num}")
    plt.xlabel("Tirage")
    plt.ylabel("Fréquence d'apparition")
    plt.title("Évolution de la fréquence d'apparition (boules principales)")
    plt.legend()
    plt.show()


def grille(data: List[Dict[str, Any]]) -> None:
    """
    Crée une grille de fréquences pour les boules principales.
    """
    verifier_donnees(data)
    logging.info("Création de la grille de fréquences pour les boules principales...")
    main_counts = Counter()
    for entry in data:
        main_counts.update(entry['main'])
    grille_freq = np.zeros((10, 10), dtype=int)
    for num, count in main_counts.items():
        x = (num - 1) // 10
        y = (num - 1) % 10
        grille_freq[x, y] = count
    plt.imshow(grille_freq, cmap='viridis')
    plt.title("Grille de fréquences d'apparition (boules principales)")
    plt.colorbar(label="Fréquence")
    plt.show()


def montecarlo(data: List[Dict[str, Any]]) -> None:
    """
    Applique une simulation Monte Carlo sur les boules principales pour estimer les résultats des tirages.
    """
    verifier_donnees(data)
    logging.info("Simulation Monte Carlo sur les boules principales...")
    main_numbers = [num for entry in data for num in entry['main']]
    total = len(main_numbers)
    counts = Counter(main_numbers)
    probs = np.array([counts[num] / total for num in sorted(counts)])
    keys = np.array(sorted(counts))
    simulations = np.random.choice(keys, size=(1000, 5), replace=True, p=probs)
    moyenne = np.mean(simulations.astype(float), axis=0)
    logging.info(f"Résultats moyens des simulations Monte Carlo (boules principales) : {moyenne}")
    alpha = 0.05
    for i in range(5):
        std_dev = np.std(simulations[:, i].astype(float))
        borne_inf, borne_sup = norm.interval(1 - alpha, loc=moyenne[i], scale=std_dev)
        logging.info(f"Intervalle de confiance pour la boule {i + 1} : [{borne_inf}, {borne_sup}]")


# ============================
# Implémentations complètes des fonctions TODO
# ============================
def theoremes_limites(data: List[Dict[str, Any]]) -> None:
    """
    Illustre le théorème central limite en affichant la distribution des moyennes d'échantillons
    calculées à partir de toutes les boules principales.
    """
    verifier_donnees(data)
    logging.info("Illustration du Théorème Central Limite (boules principales)...")
    all_main = [num for entry in data for num in entry['main']]
    all_main = np.array(all_main)
    sample_size = 30
    num_samples = 1000
    sample_means = [np.mean(np.random.choice(all_main, size=sample_size, replace=True))
                    for _ in range(num_samples)]
    sample_means = np.array(sample_means)
    plt.figure(figsize=(10, 6))
    plt.hist(sample_means, bins=30, density=True, alpha=0.6, color='g')
    mu = np.mean(all_main)
    sigma = np.std(all_main) / np.sqrt(sample_size)
    x = np.linspace(min(sample_means), max(sample_means), 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r', linewidth=2)
    plt.title("Illustration du Théorème Central Limite (boules principales)")
    plt.xlabel("Moyenne d'échantillon")
    plt.ylabel("Densité")
    plt.show()


def estimation(data: List[Dict[str, Any]]) -> None:
    """
    Effectue une estimation par ajustement d'une loi normale sur la somme des boules principales.
    """
    verifier_donnees(data)
    logging.info("Estimation des paramètres de la somme des boules principales avec une loi normale...")
    series_sum = [sum(entry['main']) for entry in data]
    mu, std = norm.fit(series_sum)
    logging.info(f"Paramètres estimés: mu = {mu:.2f}, std = {std:.2f}")
    plt.figure(figsize=(10, 6))
    plt.hist(series_sum, bins=30, density=True, alpha=0.6, color='b')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("Ajustement d'une loi normale à la somme (boules principales)")
    plt.xlabel("Somme des boules principales")
    plt.ylabel("Densité")
    plt.show()


def methodes_bayesiennes(data: List[Dict[str, Any]]) -> None:
    """
    Applique une méthode bayésienne (mise à jour Beta) pour estimer la probabilité d'apparition
    des numéros dans les boules principales.
    """
    verifier_donnees(data)
    logging.info("Application des méthodes bayésiennes pour les boules principales...")
    main_counts = Counter()
    for entry in data:
        main_counts.update(entry['main'])
    total = sum(main_counts.values())
    bayes_probs = {}
    for num, count in main_counts.items():
        alpha_post = 1 + count
        beta_post = 1 + total - count
        bayes_probs[num] = alpha_post / (alpha_post + beta_post)
    logging.info("Probabilités estimées par méthode bayésienne (boules principales) :")
    for num, prob in bayes_probs.items():
        logging.info(f"Numéro {num}: probabilité = {prob:.3f}")
    from scipy.stats import beta
    plt.figure(figsize=(12, 8))
    x_vals = np.linspace(0, 1, 100)
    for i, num in enumerate(sorted(list(main_counts.keys()))[:5]):
        count = main_counts[num]
        alpha_post = 1 + count
        beta_post = 1 + total - count
        y_vals = beta.pdf(x_vals, alpha_post, beta_post)
        plt.plot(x_vals, y_vals, label=f"Numéro {num} (α={alpha_post}, β={beta_post})")
    plt.title("Distributions Beta a posteriori (boules principales)")
    plt.xlabel("Probabilité")
    plt.ylabel("Densité")
    plt.legend()
    plt.show()


def algorithmes_specifiques(data: List[Dict[str, Any]]) -> None:
    """
    Implémente un algorithme spécifique : un clustering (K-means) des tirages enrichis,
    visualisé via PCA. Seules les caractéristiques des boules principales sont utilisées.
    """
    verifier_donnees(data)
    logging.info("Exécution d'algorithmes spécifiques: Clustering des tirages (boules principales)...")
    enriched = feature_engineering(data)
    # Extraire uniquement les features des boules principales
    features = []
    for entry in enriched:
        features.append([entry['main_sum'], entry['main_mean'], entry['main_std'],
                         entry['main_min'], entry['main_max'], entry['main_median'],
                         entry['main_range'], entry['main_ratio']])
    X = np.array(features)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    logging.info("Résultats du clustering K-means (3 clusters) :")
    for cluster_id in np.unique(clusters):
        logging.info(f"Cluster {cluster_id} : {np.sum(clusters == cluster_id)} tirages")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    for cluster_id in np.unique(clusters):
        plt.scatter(X_pca[clusters == cluster_id, 0],
                    X_pca[clusters == cluster_id, 1],
                    label=f"Cluster {cluster_id}")
    plt.title("Clustering des tirages (PCA, boules principales)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()


def simulation_tirages() -> None:
    """
    Simule des tirages indépendants :
      - 1000 tirages de 5 numéros parmi 1 à 49 pour les boules principales
      - 1000 numéros chance tirés parmi 1 à 10
    Affiche la fréquence des numéros pour chaque catégorie.
    """
    logging.info("Simulation de tirages indépendants...")
    num_tirages = 1000
    tirages_main = [sorted(random.sample(range(1, 50), 5)) for _ in range(num_tirages)]
    tirages_chance = [random.choice(range(1, 11)) for _ in range(num_tirages)]
    main_counts = Counter()
    for tirage in tirages_main:
        main_counts.update(tirage)
    chance_counts = Counter(tirages_chance)
    logging.info("Fréquence des boules principales dans les tirages simulés :")
    for num in sorted(main_counts.keys()):
        logging.info(f"Numéro {num} : {main_counts[num]} fois")
    logging.info("Fréquence du numéro chance dans les tirages simulés :")
    for num in sorted(chance_counts.keys()):
        logging.info(f"Numéro {num} : {chance_counts[num]} fois")
    plt.figure(figsize=(10, 6))
    plt.bar(list(main_counts.keys()), list(main_counts.values()), color='skyblue')
    plt.title("Fréquence d'apparition (boules principales) dans les tirages simulés")
    plt.xlabel("Numéro")
    plt.ylabel("Nombre d'apparitions")
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.bar(list(chance_counts.keys()), list(chance_counts.values()), color='salmon')
    plt.title("Fréquence d'apparition (numéro chance) dans les tirages simulés")
    plt.xlabel("Numéro")
    plt.ylabel("Nombre d'apparitions")
    plt.show()


def tests_hypotheses(data: List[Dict[str, Any]]) -> None:
    """
    Effectue des tests d'hypothèses :
      - Un test t de Student pour comparer la somme des boules principales entre la première et la deuxième moitié des tirages.
      - Un test du chi² pour vérifier l'association de la présence d'un numéro cible (dans les boules principales) entre les deux groupes.
    """
    verifier_donnees(data)
    logging.info("Tests d'hypothèses (boules principales) : Test t de Student et test du chi²...")
    series_sum = [sum(entry['main']) for entry in data]
    mid = len(series_sum) // 2
    group1 = series_sum[:mid]
    group2 = series_sum[mid:]
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    logging.info(f"Test t de Student sur la somme des boules principales : t_stat = {t_stat:.3f}, p_value = {p_val:.3f}")
    if p_val < 0.05:
        logging.info("Différence significative détectée entre les deux groupes.")
    else:
        logging.info("Aucune différence significative détectée entre les deux groupes.")
    target_number = 7
    count_group1 = sum(1 for entry in data[:mid] if target_number in entry['main'])
    count_group2 = sum(1 for entry in data[mid:] if target_number in entry['main'])
    table = np.array([[count_group1, mid - count_group1],
                      [count_group2, (len(data) - mid) - count_group2]])
    chi2, p, dof, expected = chi2_contingency(table)
    logging.info(f"Test du chi² pour le numéro {target_number} (boules principales) : chi2 = {chi2:.3f}, p = {p:.3f}")
    if p < 0.05:
        logging.info("Association significative détectée pour le numéro cible entre les groupes.")
    else:
        logging.info("Aucune association significative détectée pour le numéro cible entre les groupes.")


# ============================
# Feature Engineering Avancé
# ============================
def feature_engineering(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrichit chaque entrée avec des caractéristiques additionnelles calculées à partir des boules principales.
    Ajoute les caractéristiques suivantes pour les boules principales :
       - main_sum, main_mean, main_std, main_min, main_max, main_median, main_range, main_ratio
    Et conserve le numéro chance tel quel.
    """
    verifier_donnees(data)
    logging.info("Enrichissement des données par feature engineering avancé...")
    enriched_data = []
    for entry in data:
        main = entry['main']
        chance = entry['chance']
        new_entry = entry.copy()
        new_entry['main_sum'] = sum(main)
        new_entry['main_mean'] = np.mean(main)
        new_entry['main_std'] = np.std(main)
        new_entry['main_min'] = min(main)
        new_entry['main_max'] = max(main)
        new_entry['main_median'] = np.median(main)
        new_entry['main_range'] = new_entry['main_max'] - new_entry['main_min']
        new_entry['main_ratio'] = new_entry['main_max'] / new_entry['main_min'] if new_entry['main_min'] != 0 else np.nan
        new_entry['chance'] = chance  # Conserve le numéro chance
        enriched_data.append(new_entry)
    return enriched_data


def load_exogenous_data() -> Dict[int, Dict[str, Any]]:
    """
    Charge des données exogènes pouvant influencer les tirages.
    Pour l'instant, retourne un dictionnaire vide.
    """
    logging.info("Chargement des données exogènes... (stub)")
    return {}


# ============================
# Modèles Prédictifs
# ============================
def predictive_model_logistic(data: List[Dict[str, Any]], target_number: int = 7) -> None:
    """
    Modèle prédictif simple avec régression logistique.
    Utilise les features du tirage précédent pour prédire la présence (1) ou l'absence (0) d'un numéro cible
    parmi les boules principales.
    """
    verifier_donnees(data)
    logging.info(f"Construction du modèle prédictif (Logistic) pour le numéro cible {target_number}...")
    enriched = feature_engineering(data)
    X, y = [], []
    # Ici, les features sont issues des boules principales et du numéro chance (9 features au total)
    for i in range(1, len(enriched)):
        prev = enriched[i - 1]
        current = enriched[i]
        features = [prev['main_sum'], prev['main_mean'], prev['main_std'],
                    prev['main_min'], prev['main_max'], prev['main_median'],
                    prev['main_range'], prev['main_ratio'], prev['chance']]
        label = 1 if target_number in current['main'] else 0
        X.append(features)
        y.append(label)
    if not X:
        logging.error("Pas assez de données pour construire le modèle prédictif (Logistic).")
        return
    X = np.array(X)
    y = np.array(y)
    tscv = TimeSeriesSplit(n_splits=5)
    
    if optuna is not None:
        def objective(trial):
            C = trial.suggest_loguniform('C', 1e-2, 1e2)
            model = LogisticRegression(solver='liblinear', C=C)
            scores = []
            for train_index, test_index in tscv.split(X):
                X_train, X_val = X[train_index], X[test_index]
                y_train, y_val = y[train_index], y[test_index]
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, pred))
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        best_C = study.best_params['C']
        logging.info(f"Meilleur hyperparamètre trouvé (Optuna - Logistic) : C = {best_C}")
        best_model = LogisticRegression(solver='liblinear', C=best_C)
        best_model.fit(X, y)
    else:
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        grid = GridSearchCV(LogisticRegression(solver='liblinear'), param_grid, cv=tscv)
        grid.fit(X, y)
        best_model = grid.best_estimator_
        logging.info(f"Meilleurs hyperparamètres (GridSearch - Logistic) : {grid.best_params_}")
    
    y_pred = best_model.predict(X)
    acc = accuracy_score(y, y_pred)
    logging.info(f"Accuracy du modèle prédictif (Logistic) : {acc:.3f}")
    logging.info("Classification Report (Logistic) :")
    logging.info(f"\n{classification_report(y, y_pred)}")
    logging.info("Matrice de confusion (Logistic) :")
    logging.info(f"\n{confusion_matrix(y, y_pred)}")


def predictive_model_ensemble(data: List[Dict[str, Any]], target_number: int = 7) -> None:
    """
    Modèle prédictif utilisant un classifieur d'ensemble (RandomForest) calibré pour prédire la présence
    d'un numéro cible parmi les boules principales.
    """
    verifier_donnees(data)
    logging.info(f"Construction du modèle prédictif (Ensemble) pour le numéro cible {target_number}...")
    enriched = feature_engineering(data)
    X, y = [], []
    for i in range(1, len(enriched)):
        prev = enriched[i - 1]
        current = enriched[i]
        features = [prev['main_sum'], prev['main_mean'], prev['main_std'],
                    prev['main_min'], prev['main_max'], prev['main_median'],
                    prev['main_range'], prev['main_ratio'], prev['chance']]
        label = 1 if target_number in current['main'] else 0
        X.append(features)
        y.append(label)
    if not X:
        logging.error("Pas assez de données pour construire le modèle prédictif (Ensemble).")
        return
    X = np.array(X)
    y = np.array(y)
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=tscv)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    calibrated_model = CalibratedClassifierCV(best_model, cv=tscv)
    calibrated_model.fit(X, y)
    logging.info(f"Meilleurs hyperparamètres (Ensemble) : {grid.best_params_}")
    y_pred = calibrated_model.predict(X)
    acc = accuracy_score(y, y_pred)
    logging.info(f"Accuracy du modèle prédictif (Ensemble) : {acc:.3f}")
    logging.info("Classification Report (Ensemble) :")
    logging.info(f"\n{classification_report(y, y_pred)}")
    logging.info("Matrice de confusion (Ensemble) :")
    logging.info(f"\n{confusion_matrix(y, y_pred)}")


def predictive_model_stacking(data: List[Dict[str, Any]], target_number: int = 7) -> None:
    """
    Modèle prédictif par stacking combinant plusieurs classifieurs pour prédire la présence
    d'un numéro cible parmi les boules principales.
    """
    verifier_donnees(data)
    logging.info(f"Construction du modèle prédictif (Stacking) pour le numéro cible {target_number}...")
    enriched = feature_engineering(data)
    X, y = [], []
    for i in range(1, len(enriched)):
        prev = enriched[i - 1]
        current = enriched[i]
        features = [prev['main_sum'], prev['main_mean'], prev['main_std'],
                    prev['main_min'], prev['main_max'], prev['main_median'],
                    prev['main_range'], prev['main_ratio'], prev['chance']]
        label = 1 if target_number in current['main'] else 0
        X.append(features)
        y.append(label)
    if not X:
        logging.error("Pas assez de données pour construire le modèle prédictif (Stacking).")
        return
    X = np.array(X)
    y = np.array(y)
    
    estimators = [
        ('lr', LogisticRegression(solver='liblinear', C=1.0)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(solver='liblinear'))
    stacking_model.fit(X, y)
    y_pred = stacking_model.predict(X)
    acc = accuracy_score(y, y_pred)
    logging.info(f"Accuracy du modèle prédictif (Stacking) : {acc:.3f}")
    logging.info("Classification Report (Stacking) :")
    logging.info(f"\n{classification_report(y, y_pred)}")
    logging.info("Matrice de confusion (Stacking) :")
    logging.info(f"\n{confusion_matrix(y, y_pred)}")


def predictive_model_deep_learning(data: List[Dict[str, Any]], target_number: int = 7, epochs: int = 50, batch_size: int = 16) -> None:
    """
    Modèle prédictif utilisant un réseau de neurones feed-forward pour prédire la présence
    d'un numéro cible parmi les boules principales.
    """
    verifier_donnees(data)
    logging.info(f"Construction du modèle prédictif (Deep Learning FF) pour le numéro cible {target_number}...")
    enriched = feature_engineering(data)
    X, y = [], []
    for i in range(1, len(enriched)):
        prev = enriched[i - 1]
        current = enriched[i]
        features = [prev['main_sum'], prev['main_mean'], prev['main_std'],
                    prev['main_min'], prev['main_max'], prev['main_median'],
                    prev['main_range'], prev['main_ratio'], prev['chance']]
        label = 1 if target_number in current['main'] else 0
        X.append(features)
        y.append(label)
    if not X:
        logging.error("Pas assez de données pour construire le modèle prédictif (Deep Learning FF).")
        return
    X = np.array(X)
    y = np.array(y)
    split_idx = int(0.7 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Accuracy du modèle prédictif (Deep Learning FF) : {acc:.3f}")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    logging.info("Classification Report (Deep Learning FF) :")
    logging.info(f"\n{classification_report(y_test, y_pred)}")
    logging.info("Matrice de confusion (Deep Learning FF) :")
    logging.info(f"\n{confusion_matrix(y_test, y_pred)}")


def predictive_model_lstm(data: List[Dict[str, Any]], target_number: int = 7, window_size: int = 3, epochs: int = 50, batch_size: int = 16) -> None:
    """
    Modèle prédictif utilisant un réseau de neurones LSTM pour capturer les dynamiques temporelles
    et prédire la présence d'un numéro cible parmi les boules principales.
    Utilise une fenêtre glissante sur les tirages.
    """
    verifier_donnees(data)
    logging.info(f"Construction du modèle prédictif (LSTM) pour le numéro cible {target_number} avec une fenêtre de taille {window_size}...")
    enriched = feature_engineering(data)
    X_seq, y_seq = [], []
    for i in range(window_size, len(enriched)):
        sequence = []
        for j in range(i - window_size, i):
            entry = enriched[j]
            features = [entry['main_sum'], entry['main_mean'], entry['main_std'],
                        entry['main_min'], entry['main_max'], entry['main_median'],
                        entry['main_range'], entry['main_ratio'], entry['chance']]
            sequence.append(features)
        label = 1 if target_number in enriched[i]['main'] else 0
        X_seq.append(sequence)
        y_seq.append(label)
    if not X_seq:
        logging.error("Pas assez de données pour construire le modèle prédictif (LSTM).")
        return
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    split_idx = int(0.7 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    nsamples, ntimesteps, nfeatures = X_train.shape
    X_train = X_train.reshape(-1, nfeatures)
    X_test = X_test.reshape(-1, nfeatures)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = X_train.reshape(nsamples, ntimesteps, nfeatures)
    X_test = X_test.reshape(-1, ntimesteps, nfeatures)
    model = Sequential([
        LSTM(64, input_shape=(ntimesteps, nfeatures), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Accuracy du modèle prédictif (LSTM) : {acc:.3f}")
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    logging.info("Classification Report (LSTM) :")
    logging.info(f"\n{classification_report(y_test, y_pred)}")
    logging.info("Matrice de confusion (LSTM) :")
    logging.info(f"\n{confusion_matrix(y_test, y_pred)}")


# ============================
# Fonction main et gestion des arguments
# ============================
def main() -> None:
    # Données intégrées compressées (gzip, encodées en Base64)
    # ATTENTION : La chaîne ci-dessous doit être remplacée par les données réelles,
    # chaque entrée devant être un dictionnaire avec les clés "main" et "chance".
    historique_loto_compressed = """
H4sIAAAAAAAAA+3SMQqDMAwF0L9xFIfU9OjDhtESVViKlBhG9HeLpGQFQ9F9+nSdx5FuaRxzrJMr
... (données complètes attendues ici)
"""
    historique_loto = charger_historique_loto(historique_loto_compressed)
    if not historique_loto:
        logging.error("Erreur : Impossible de charger les données. Le programme va s'arrêter.")
        return
    # Enrichir les données via feature engineering avancé
    historique_loto = feature_engineering(historique_loto)
    parser = argparse.ArgumentParser(description="Programme de statistique et de prédiction pour la loterie française")
    parser.add_argument("--analyse", type=str, default="stats",
                        help=("Type d'analyse à exécuter: stats, simulation, montecarlo, arima, predict, "
                              "theoremes, estimation, bayesian, algos, sim_tirages, tests"))
    parser.add_argument("--predictor", type=str, default="logistic",
                        help=("Modèle prédictif à utiliser (si --analyse predict) : "
                              "logistic, logistic_bayesian, ensemble, stacking, deep, lstm"))
    parser.add_argument("--target", type=int, default=7,
                        help="Numéro cible pour le modèle prédictif (par défaut 7)")
    args = parser.parse_args()

    if args.analyse == "stats":
        mesures = calculer_mesures_statistiques(historique_loto)
        logging.info(f"Mesures statistiques : {mesures}")
    elif args.analyse == "simulation":
        machine(historique_loto)
    elif args.analyse == "montecarlo":
        montecarlo(historique_loto)
    elif args.analyse == "arima":
        # Par exemple, appliquer ARIMA sur la somme des boules principales
        series_sum = [sum(entry['main']) for entry in historique_loto]
        try:
            model = ARIMA(series_sum, order=(1, 0, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            logging.info(f"Prévision ARIMA pour la somme des boules principales du prochain tirage : {forecast[0]}")
        except Exception as e:
            logging.error(f"Erreur lors de l'ajustement ARIMA : {e}")
    elif args.analyse == "predict":
        if args.predictor == "logistic":
            predictive_model_logistic(historique_loto, target_number=args.target)
        elif args.predictor == "logistic_bayesian":
            predictive_model_logistic(historique_loto, target_number=args.target)
        elif args.predictor == "ensemble":
            predictive_model_ensemble(historique_loto, target_number=args.target)
        elif args.predictor == "stacking":
            predictive_model_stacking(historique_loto, target_number=args.target)
        elif args.predictor == "deep":
            predictive_model_deep_learning(historique_loto, target_number=args.target)
        elif args.predictor == "lstm":
            predictive_model_lstm(historique_loto, target_number=args.target)
        else:
            logging.warning("Modèle prédictif non reconnu. Utilisation du modèle logistic par défaut.")
            predictive_model_logistic(historique_loto, target_number=args.target)
    elif args.analyse == "theoremes":
        theoremes_limites(historique_loto)
    elif args.analyse == "estimation":
        estimation(historique_loto)
    elif args.analyse == "bayesian":
        methodes_bayesiennes(historique_loto)
    elif args.analyse == "algos":
        algorithmes_specifiques(historique_loto)
    elif args.analyse == "sim_tirages":
        simulation_tirages()
    elif args.analyse == "tests":
        tests_hypotheses(historique_loto)
    else:
        logging.warning("Type d'analyse non reconnu. Exécution par défaut de l'analyse statistique.")
        mesures = calculer_mesures_statistiques(historique_loto)
        logging.info(f"Mesures statistiques : {mesures}")


if __name__ == "__main__":
    main()
