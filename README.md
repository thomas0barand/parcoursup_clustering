# Branche Bastien - Génération Systématique de Visualisations

Scripts pour générer automatiquement des visualisations comparatives des résultats de clustering multi-algorithmes.

## Caractéristiques

- **Visualisations comparatives** : Boxplots et distributions par cluster pour K-means, DBSCAN, clustering hiérarchique
- **Génération automatique** : Script `generate_cluster_plots.py` pour produire des visualisations systématiques
- **Multi-algorithmes** : Comparaison des résultats entre différentes méthodes de clustering
- **Organisation** : Visualisations organisées par dataset et caractéristiques

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Génération des visualisations

```bash
python generate_cluster_plots.py
```

Génère automatiquement des boxplots pour chaque feature numérique, organisés par dataset de clustering.

### Datasets analysés

Le script traite par défaut :
- `data/df_clustering_kmeans_3.csv`
- `data/df_clustering_dbscan_3.5_6.csv`
- `data/df_hierarchical_clusters_3.csv`

### Structure des visualisations

Les résultats sont organisés dans `visualization_images/[dataset_name]/` avec :
- Boxplots par feature montrant la distribution par cluster
- Moyennes affichées pour chaque groupe
- Format standardisé pour comparaison

## Configuration

Modifiez `DATASETS` dans `generate_cluster_plots.py` pour ajouter/retirer des datasets :

```python
DATASETS = [
    ("data/df_clustering_kmeans_3.csv", "kmeans_3"),
    ("data/df_clustering_dbscan_3.5_6.csv", "dbscan_3.5_6"),
    ("data/df_hierarchical_clusters_3.csv", "hierarchical_3"),
]
```

## Résultats

- `visualization_images/[dataset]/[feature].png` : Boxplots par feature et par dataset
- Visualisations organisées pour comparaison rapide entre algorithmes

## Stack

Python 3.x, pandas, numpy, matplotlib

---

**Auteur** : Thomas Barand | Projet Data Mining - Parcoursup 2024
