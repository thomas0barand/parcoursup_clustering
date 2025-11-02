# Branche Thomas - Pipeline de Clustering CPGE

Pipeline complet de clustering pour l'analyse des CPGE avec évaluation systématique des paramètres et visualisations avancées.

## Caractéristiques

- **Algorithmes** : K-means, DBSCAN, Clustering hiérarchique
- **Évaluation** : Grille de recherche DBSCAN avec métriques (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- **Visualisations** : PCA interactives, distributions par cluster, heatmaps, dendrogrammes
- **Normalisation** : Support robust/standard/minmax avec contrôle poids géographique

## Configuration

- **Filière** : CPGE (toutes sous-filières)
- **Algorithme** : DBSCAN (eps=3.0, min_samples=6)
- **Normalisation** : Robust Scaler, alpha_geographique=0.965

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Pipeline complet
```bash
python main.py
```
Génère : features normalisées → clustering DBSCAN → visualisations → métriques → résultats CSV

### Configuration (`main.py`)
```python
filieres1 = ["CPGE"]           # Filières à analyser
normalize_method = 'robust'     # 'robust', 'standard', 'minmax'
alpha_geographique = 0.965     # Poids données géographiques
eps = 3.0                      # Paramètres DBSCAN
min_samples = 6
```

### Analyses
```bash
jupyter notebook notebooks/eda_parcoursup.ipynb  # Exploration
python hierarchical.py                           # Clustering hiérarchique
```

## Features

**Numériques** : sélectivité (taux d'accès, rang, mentions), caractéristiques sociales (boursiers, géographie), demande/capacité (ratios, capacité), profil admis (type bac, mentions)

**Catégorielles** : filière détaillée, statut établissement

## Résultats

- `data/df_clustering_dbscan_*.csv` : Assignations clusters
- `images/clustering_vis/` : Visualisations PCA et distributions
- `images/clustering_vis/silhouette_dbscan_heatmap.png` : Heatmap métriques

## Stack

Python 3.12+, scikit-learn, scipy, pandas, numpy, matplotlib/seaborn/plotly

---

**Auteur** : Thomas Barand | Projet Data Mining - Parcoursup 2024
