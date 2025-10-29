# Étude de l'Élitisme dans les Écoles d'Ingénieurs et CPGE

## Problématique

**Peut-on identifier des groupes naturels de formations au sein des écoles d'ingénieurs et des CPGE selon leurs caractéristiques d'élitisme ?**

Cette étude vise à analyser les données Parcoursup 2024 pour comprendre comment se structure l'élitisme dans l'enseignement supérieur français, en se concentrant sur les écoles d'ingénieurs et les classes préparatoires (CPGE).

## Source des Données

- **Dataset**: fr-esr-parcoursup.csv
- **Année**: 2024
- **Source**: Ministère de l'Enseignement Supérieur et de la Recherche
- **Focus**: Écoles d'ingénieurs et CPGE

## Méthodologie

L'analyse se déroule en **trois phases** :

### Phase 1 : Exploration des Données
- Exploration globale du dataset Parcoursup
- Extraction et analyse des écoles d'ingénieurs et CPGE
- Statistiques descriptives et visualisations
- Identification des variables clés

### Phase 2 : Construction des Features d'Élitisme
Définition et création de variables mesurant l'élitisme :
- **Sélectivité académique** : mentions au bac, taux d'accès, rang dernier appelé
- **Sélectivité sociale** : % de boursiers, origine géographique
- **Prestige** : ratio candidats/places, capacité d'accueil
- **Profil des admis** : type de bac, mentions obtenues

### Phase 3 : Clustering et Analyse
Application de plusieurs algorithmes de clustering :
- **K-means** : pour identifier des groupes homogènes
- **DBSCAN** : pour détecter les outliers (formations atypiques)
- **Clustering hiérarchique** : pour visualiser la hiérarchie de l'élitisme

Validation avec métriques (silhouette, inertie) et interprétation des clusters identifiés.

## Technologies

- Python 3.x
- pandas, numpy, scikit-learn
- matplotlib, seaborn, plotly

## Structure

```
parcoursup/
├── data/
│   └── fr-esr-parcoursup.csv
├── notebooks/
│   └── eda_parcoursup.ipynb
├── README.md
└── requirements.txt
```

## Utilisation

1. Installer les dépendances : `pip install -r requirements.txt`
2. Lancer le notebook : `jupyter notebook notebooks/eda_parcoursup.ipynb`
3. Suivre les 3 phases d'analyse

---

**Auteur** : Thomas  
**Date** : Octobre 2024  
**Cadre** : Projet Data Mining - Analyse Parcoursup

