# Analyse de l'Élitisme dans les Écoles d'Ingénieurs et les CPGE

## Problématique

**Dans quelle mesure les écoles d'ingénieurs et les Classes Préparatoires aux Grandes Écoles (CPGE) scientifiques forment-elles des groupes distincts selon leurs caractéristiques d'élitisme ?**

Cette question, au cœur de notre étude, vise à décrypter la structuration de ces formations d'excellence de l'enseignement supérieur français. À travers l'analyse des données Parcoursup 2024, nous explorons comment ces établissements se différencient selon plusieurs dimensions de l'élitisme : sélectivité académique, diversité sociale, géographique et de genre.

Notre démarche mobilise des techniques d'analyse exploratoire et de clustering pour dépasser les classifications traditionnelles (public/privé, Paris/province) en combinant différents indicateurs : taux d'accès, proportion de boursiers, diversité des mentions au bac, mixité et origines géographiques des admis. L'identification de "profils types" émergents pourrait non seulement éclairer les mécanismes de reproduction sociale à l'œuvre dans ces filières, mais aussi contribuer à la réflexion sur des politiques éducatives plus inclusives.

## Source des Données

- **Dataset** : `data/fr-esr-parcoursup.csv`
- **Année** : 2024
- **Source** : Ministère de l'Enseignement Supérieur et de la Recherche
- **Focus** : Écoles d'ingénieurs et CPGE scientifiques

## Structure du Répertoire

```
parcoursup/
├── data/                           # Données brutes et traitées
│   ├── fr-esr-parcoursup.csv       # Dataset source Parcoursup 2024
│   ├── df_features.csv              # Features extraites
│   ├── df_features_cleaned.csv     # Features nettoyées
│   ├── df_features_normalized.csv  # Features normalisées pour clustering
│   └── df_clustering_*.csv         # Résultats de clustering (k-means, DBSCAN, etc.)
│
├── images/                         # Visualisations générées
│   ├── clustering_vis/            # Visualisations de clustering (PCA, distributions, etc.)
│   ├── hierarchical_*.png          # Dendrogrammes et analyses hiérarchiques
│   └── correlation_matrix.png     # Matrice de corrélation
│
├── notebooks/                      # Notebooks d'analyse exploratoire
│   ├── eda_parcoursup.ipynb        # Analyse exploratoire principale
│   └── eda.ipynb                   # Analyses exploratoires additionnelles
│
├── features.py                     # Extraction et préparation des features d'élitisme
├── clustering.py                   # Implémentation des algorithmes de clustering (K-means, DBSCAN)
├── hierarchical.py                 # Clustering hiérarchique et évaluation des métriques de distance
├── visualize.py                    # Fonctions de visualisation
├── main.py                         # Script principal d'exécution du pipeline
│
├── requirements.txt                # Dépendances Python
├── pyproject.toml                  # Configuration Poetry (optionnel)
└── README.md                       # Ce fichier
```

## Installation

### Prérequis
- Python 3.12+
- pip ou Poetry (pour la gestion des dépendances)

### Installation des dépendances

**Option 1 : Avec pip**
```bash
pip install -r requirements.txt
```

**Option 2 : Avec Poetry**
```bash
poetry install
```

Les principales dépendances incluent :
- `pandas`, `numpy` : Manipulation de données
- `scikit-learn` : Clustering et machine learning
- `matplotlib`, `seaborn`, `plotly` : Visualisation
- `scipy` : Calculs statistiques et clustering hiérarchique

## Utilisation

### Configuration Actuelle (Branche `thomas`)

⚠️ **Note** : Ce README et les fonctionnalités décrites ci-dessous sont spécifiques à la branche `thomas` pour le moment.

### Pipeline d'Analyse

Le workflow principal suit trois étapes :

#### 1. Extraction et Préparation des Features

Les features d'élitisme sont extraites et nettoyées depuis les données brutes :

```python
# Voir features.py pour les détails
from features import load_raw_data, clean_data, normalize_features_for_clustering

# Chargement des données filtrées (CPGE uniquement dans la config actuelle)
df_final = load_raw_data("data/fr-esr-parcoursup.csv", filieres1=["CPGE"], filieres2=[None])

# Nettoyage (alpha_geographique contrôle le poids des données géographiques)
df_final_cleaned = clean_data(df_final, alpha_geographique=0.965)

# Normalisation (robust, standard, ou minmax)
df_normalized, scaler, numeric_cols = normalize_features_for_clustering(
    df_final_cleaned, 
    exclude_cols=categorical_cols,
    method='robust'  # ou 'standard', 'minmax'
)
```

**Features d'élitisme extraites** :
- **Sélectivité académique** : taux d'accès, rang du dernier appelé, mentions au bac
- **Sélectivité sociale** : proportion de boursiers, origine géographique
- **Prestige** : ratio candidats/places, capacité d'accueil
- **Profil des admis** : type de bac, distribution des mentions

#### 2. Clustering

Plusieurs algorithmes sont disponibles :

**K-means** :
```python
from clustering import perform_clustering
kmeans, labels = perform_clustering(X, n_clusters=3, method='kmeans')
```

**DBSCAN** (détection d'outliers) :
```python
dbscan, labels = perform_clustering(
    X, 
    method='dbscan', 
    eps=3.0, 
    min_samples=6
)
```

**Clustering Hiérarchique** :
```bash
python hierarchical.py
```
Permet d'évaluer différentes métriques de distance et génère des dendrogrammes.

#### 3. Visualisation et Analyse

Les visualisations sont générées automatiquement :
- Projections PCA avec coloration par clusters
- Distributions des features par cluster
- Dendrogrammes pour le clustering hiérarchique
- Heatmaps de corrélation et métriques de clustering

### Exécution Complète

Pour exécuter le pipeline complet tel que configuré actuellement :

```bash
python main.py
```

Le script `main.py` :
1. Charge et nettoie les données
2. Normalise les features
3. Effectue le clustering DBSCAN (configuré pour eps=3.0, min_samples=6)
4. Génère les visualisations et métriques
5. Sauvegarde les résultats dans `data/df_clustering_dbscan_*.csv`

### Analyse Exploratoire

Pour explorer les données de manière interactive :

```bash
jupyter notebook notebooks/eda_parcoursup.ipynb
```

Les notebooks contiennent :
- Statistiques descriptives
- Analyses de corrélation
- Visualisations exploratoires
- Tests de différentes configurations de clustering

## Paramètres Configurables

### Dans `main.py` :

- **`filieres1`** : Filières très agrégées à analyser (ex: `["CPGE"]`, `["Ecole d'Ingénieur"]`)
- **`filieres2`** : Filières détaillées correspondantes (ex: `[None]` pour toutes)
- **`alpha_geographique`** : Poids des données géographiques (0.965 par défaut)
- **`normalize_method`** : Méthode de normalisation (`'robust'`, `'standard'`, `'minmax'`)
- **`n_clusters`** : Nombre de clusters pour K-means (3 par défaut)
- **`eps`**, **`min_samples`** : Paramètres DBSCAN

### Features Utilisées

Les features numériques et catégorielles sont définies dans `main.py` :

- **Numériques** : ratios de candidats/admis, sélectivité, mentions au bac, données géographiques, etc.
- **Catégorielles** : filière détaillée, statut établissement (public/privé)

## Résultats et Interprétation

Les résultats sont sauvegardés dans :
- **`data/df_clustering_*.csv`** : Datasets avec colonne `cluster` ajoutée
- **`images/clustering_vis/`** : Visualisations PCA et distributions par cluster
- **`images/hierarchical_*.png`** : Dendrogrammes et analyses hiérarchiques

### Métriques de Validation

Les métriques calculées incluent :
- **Silhouette Score** : Mesure de cohésion et séparation des clusters
- **Calinski-Harabasz Index** : Ratio entre variance inter-cluster et intra-cluster
- **Davies-Bouldin Index** : Mesure de similarité entre clusters (plus bas = mieux)

## Technologies Utilisées

- **Python 3.12+**
- **pandas, numpy** : Manipulation de données
- **scikit-learn** : Clustering (K-means, DBSCAN, AgglomerativeClustering)
- **scipy** : Clustering hiérarchique, métriques de distance
- **matplotlib, seaborn, plotly** : Visualisation
- **Jupyter** : Analyse exploratoire interactive

## Auteur

**Thomas Barand**  
Projet Data Mining - Analyse Parcoursup 2024  
Octobre 2024

---

## Notes sur cette Branche

Cette branche (`thomas`) est actuellement configurée pour analyser spécifiquement les **CPGE** avec une configuration DBSCAN optimisée. Les paramètres peuvent être modifiés dans `main.py` pour analyser d'autres filières ou ajuster les algorithmes de clustering.
