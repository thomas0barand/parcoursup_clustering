from features import load_raw_data, clean_data, normalize_features_for_clustering
from clustering import load_and_prepare_data, perform_clustering, print_metrics, perform_pca, create_interactive_plot, create_static_visualizations
from clustering import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
import pandas as pd

categorical_cols = ["Code UAI de l'établissement", 
                   "Établissement",
                   "Statut de l’établissement de la filière de formation (public, privé…)",
                   "Département de l'établissement",
                   "Région de l'établissement",
                   "Commune de l'établissement",
                   "Coordonnées GPS de la formation",
                   "Filière de formation",
                   "Filière de formation très agrégée",
                   "Filière de formation détaillée bis"]

CATEGORICAL_FEATURES = [
    # "Département de l’établissement",
    # "Filière de formation très agrégée",
    "Filière de formation détaillée bis"
    # "Statut de l’établissement de la filière de formation (public, privé…)"
]

NUMERICAL_FEATURES = [
    'Rang du dernier appelé du groupe 1',
    'Capacité de l’établissement par formation',
    'Effectif total des candidats en phase principale',
    'f_ratio_candidats',
    'f_ratio_admis',
    'f_selectivity_candidats',
    # 'b_ratio_candidats',
    # 'b_ratio_admis',
    # 'b_selectivity_candidats',
    # 'gen_candidats_ratio',
    # 'tech_candidats_ratio',
    # 'prof_candidats_ratio',
    # 'gen_admis_ratio',
    # 'tech_admis_ratio',
    # 'prof_admis_ratio',
    # 'gen_selectivity_candidats',
    # 'tech_selectivity_candidats',
    # 'prof_selectivity_candidats',
    'sans_mention_ratio',
    'assez_bien_mention_ratio',
    'bien_mention_ratio',
    'tres_bien_mention_ratio',
    'tres_bien_avec_felicitation_mention_ratio',
    'meme_academie_ratio',
    'meme_etablissement_ratio',
    'last_call_rank_ratio',
    'pressure_ratio',
    'taux_acces_ratio',
    'longitude',
    'latitude'
]

# filieres1 = ["Ecole d'Ingénieur", "CPGE", "BTS", "BUT", "Licence", "Licence_Las", "IFSI", "PASS", "EFTS", "Ecole de Commerce", "Autre Formation"]
# filieres2 = [None for _ in range(len(filieres1))]

filieres1 = ["CPGE"]
filieres2= [None]

normalize_method = 'robust'
n_clusters = 2


## NORMALISATION DES FEATURES
df_final = load_raw_data("data/fr-esr-parcoursup.csv", filieres1, filieres2)

# alpha_geographique donne la valmeur que l'on donne au données géographiques (1.00 : les données géo sont très représentées, <0.9 : les données géo sont très peu représentées)
df_final_cleaned = clean_data(df_final, alpha_geographique=0.90)

df_normalized, scaler, numeric_cols_normalized = normalize_features_for_clustering(
    df_final_cleaned, 
    exclude_cols=categorical_cols,
    method=normalize_method
)

df_normalized.to_csv("data/df_features_normalized.csv", index=False)


## CLUSTERING
df_clustering, X = load_and_prepare_data("data/df_features_normalized.csv", NUMERICAL_FEATURES, CATEGORICAL_FEATURES)



# Perform clustering
kmeans, labels = perform_clustering(X, n_clusters=n_clusters)
df_clustering['cluster'] = labels

# Save results
df_clustering.to_csv('data/df_clustering.csv', index=False)

# Print metrics
print_metrics(X, labels, kmeans)

# Perform PCA
pca, X_pca = perform_pca(X)

# Create visualizations
create_interactive_plot(df_clustering, X_pca, pca)
create_static_visualizations(df_clustering, X, X_pca, pca, kmeans)