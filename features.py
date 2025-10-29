# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings('ignore')

# configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

def load_raw_data(data_path='data/fr-esr-parcoursup.csv', 
                  filiere_tres_agregee=None, 
                  filiere_detaillee=None):
    """
    Charge et filtre les données Parcoursup.
    
    Parameters:
    -----------
    data_path : str
        Chemin vers le fichier CSV
    filiere_tres_agregee : list[str], optional
        Liste des filières très agrégées à filtrer (ex: ["Ecole d'Ingénieur", "CPGE"])
        Par défaut: ["Ecole d'Ingénieur", "CPGE"]
    filiere_detaillee : list[str], optional
        Liste des filières détaillées correspondantes (doit avoir la même longueur que filiere_tres_agregee)
        Mettre None pour une filière si on veut toutes les sous-filières
        ex: [None, "Classe préparatoire scientifique"] signifie:
            - Toutes les "Ecole d'Ingénieur"
            - Seulement les CPGE de type "Classe préparatoire scientifique"
        Par défaut: [None, "Classe préparatoire scientifique"]
    """
    # Définir les valeurs par défaut à l'intérieur de la fonction (évite les problèmes de mutable default arguments)
    if filiere_tres_agregee is None:
        filiere_tres_agregee = ["Ecole d'Ingénieur", "CPGE"]
    if filiere_detaillee is None:
        filiere_detaillee = [None, "Classe préparatoire scientifique"]
    
    # Chargement des données 
    df = pd.read_csv(data_path, sep=';', low_memory=False)
    
    # Construire le filtre dynamiquement
    print("\n" + "="*80)
    print("FILTRAGE DES DONNÉES")
    print("="*80)
    print(f"Filières très agrégées: {filiere_tres_agregee}")
    print(f"Filières détaillées: {filiere_detaillee}")
    
    if len(filiere_tres_agregee) != len(filiere_detaillee):
        raise ValueError("Les listes filiere_tres_agregee et filiere_detaillee doivent avoir la même longueur")
    
    # Construire les conditions de filtrage
    conditions = []
    for filiere_main, filiere_sub in zip(filiere_tres_agregee, filiere_detaillee):
        if filiere_sub is None:
            # Pas de filtre sur la sous-filière, prendre toutes les formations de cette filière principale
            condition = (df['Filière de formation très agrégée'] == filiere_main)
            print(f"  ✓ Inclus: TOUTES les formations '{filiere_main}'")
        else:
            # Filtre sur la filière principale ET la sous-filière
            condition = (
                (df['Filière de formation très agrégée'] == filiere_main) &
                (df['Filière de formation.1'] == filiere_sub)
            )
            print(f"  ✓ Inclus: '{filiere_main}' avec sous-filière '{filiere_sub}'")
        conditions.append(condition)
    
    # Combiner toutes les conditions avec OR
    final_condition = conditions[0]
    for condition in conditions[1:]:
        final_condition = final_condition | condition
    
    df_inge = df[final_condition].copy()
    
    print(f"\n✓ {len(df_inge)} formations filtrées sur {len(df)} total ({100*len(df_inge)/len(df):.2f}%)")
    print("="*80)

    FEATURES_KEYS = ["Code UAI de l'établissement",
    "Établissement"]

    FEATURES_CLASSIFICATION = ["Statut de l’établissement de la filière de formation (public, privé…)",
    "Département de l’établissement",
    "Région de l’établissement", 
    "Commune de l’établissement",
    "Coordonnées GPS de la formation",
    "Filière de formation",
    "Filière de formation très agrégée",
    "Filière de formation détaillée bis",
    "Capacité de l’établissement par formation",    
    "Effectif total des candidats en phase principale",
    "Rang du dernier appelé du groupe 1"
]

    ## Calcul des fetures liées au genre
    f_ratio_candidats = df_inge["Dont effectif des candidates pour une formation"] / df_inge["Effectif total des candidats en phase principale"]
    f_ratio_admis = df_inge["% d’admis dont filles"]/100

    f_selectivity_candidats = f_ratio_candidats / f_ratio_admis

    ## Calcul des features liées aux boursiers

    candidats_boursiers = (df_inge["Dont effectif des candidats boursiers néo bacheliers généraux en phase principale"] +
    df_inge["Dont effectif des candidats boursiers néo bacheliers technologiques en phase principale"] +
    df_inge["Dont effectif des candidats boursiers néo bacheliers professionnels en phase principale"])


    b_ratio_candidats = candidats_boursiers / df_inge["Effectif total des candidats en phase principale"]
    b_ratio_admis = df_inge["% d’admis néo bacheliers boursiers"]/100
    b_selectivity_candidats = b_ratio_candidats / b_ratio_admis

    ## Calcul de features liées à l'origine (quel type bac)
    gen_candidats_ratio = df_inge["Effectif des candidats néo bacheliers généraux en phase principale"] / df_inge["Effectif total des candidats en phase principale"]
    tech_candidats_ratio = df_inge["Effectif des candidats néo bacheliers technologiques en phase principale"] / df_inge["Effectif total des candidats en phase principale"]
    prof_candidats_ratio = df_inge["Effectif des candidats néo bacheliers professionnels en phase principale"] / df_inge["Effectif total des candidats en phase principale"]

    gen_admis_ratio = df_inge["% d’admis néo bacheliers généraux"]/100
    tech_admis_ratio = df_inge["% d’admis néo bacheliers technologiques"]/100
    prof_admis_ratio = df_inge["% d’admis néo bacheliers professionnels"]/100


    gen_selectivity_candidats = gen_candidats_ratio / gen_admis_ratio
    tech_selectivity_candidats = tech_candidats_ratio / tech_admis_ratio
    prof_selectivity_candidats = prof_candidats_ratio / prof_admis_ratio

    ## Calcul de fetaures liées à la mention au bac
    sans_mention_ratio = df_inge["% d’admis néo bacheliers sans mention au bac"]/100
    assez_bien_mention_ratio = df_inge["% d’admis néo bacheliers avec mention Assez Bien au bac"]/100
    bien_mention_ratio = df_inge["% d’admis néo bacheliers avec mention Bien au bac"]/100
    tres_bien_mention_ratio = df_inge["% d’admis néo bacheliers avec mention Très Bien au bac"]/100
    tres_bien_avec_felicitation_mention_ratio = df_inge["% d’admis néo bacheliers avec mention Très Bien avec félicitations au bac"]/100


    ## Calcul de features liées à l'origine des candidats et admis

    meme_academie_ratio = df_inge["% d’admis néo bacheliers issus de la même académie"]/100
    meme_etablissement_ratio = df_inge["% d’admis néo bacheliers issus du même établissement (BTS/CPGE)"]/100

    ## Calcul de features liées au rang du dernier appelé, à l'estimation de la part de refus, à la selectivite (features "plus hautes")

    last_call_rank_ratio = (df_inge["Rang du dernier appelé du groupe 1"] - df_inge["Capacité de l’établissement par formation"])/ df_inge["Effectif total des candidats en phase principale"]
    pressure_ratio = df_inge["Capacité de l’établissement par formation"]/df_inge["Effectif total des candidats en phase principale"]
    taux_acces_ratio = df_inge["Taux d’accès"]/100


    # Création du dataframe final avec les features sélectionnées et créées
    df_final = pd.DataFrame()

    # Ajouter les features originales (CAPITAL)
    for feature in FEATURES_KEYS + FEATURES_CLASSIFICATION:
        df_final[feature] = df_inge[feature]

    # Ajouter les features créées
    df_final['f_ratio_candidats'] = f_ratio_candidats
    df_final['f_ratio_admis'] = f_ratio_admis
    df_final['f_selectivity_candidats'] = f_selectivity_candidats

    df_final['b_ratio_candidats'] = b_ratio_candidats
    df_final['b_ratio_admis'] = b_ratio_admis
    df_final['b_selectivity_candidats'] = b_selectivity_candidats

    df_final['gen_candidats_ratio'] = gen_candidats_ratio
    df_final['tech_candidats_ratio'] = tech_candidats_ratio
    df_final['prof_candidats_ratio'] = prof_candidats_ratio

    df_final['gen_admis_ratio'] = gen_admis_ratio
    df_final['tech_admis_ratio'] = tech_admis_ratio
    df_final['prof_admis_ratio'] = prof_admis_ratio

    df_final['gen_selectivity_candidats'] = gen_selectivity_candidats
    df_final['tech_selectivity_candidats'] = tech_selectivity_candidats
    df_final['prof_selectivity_candidats'] = prof_selectivity_candidats

    df_final['sans_mention_ratio'] = sans_mention_ratio
    df_final['assez_bien_mention_ratio'] = assez_bien_mention_ratio
    df_final['bien_mention_ratio'] = bien_mention_ratio
    df_final['tres_bien_mention_ratio'] = tres_bien_mention_ratio
    df_final['tres_bien_avec_felicitation_mention_ratio'] = tres_bien_avec_felicitation_mention_ratio

    df_final['meme_academie_ratio'] = meme_academie_ratio
    df_final['meme_etablissement_ratio'] = meme_etablissement_ratio

    df_final['last_call_rank_ratio'] = last_call_rank_ratio
    df_final['pressure_ratio'] = pressure_ratio
    df_final['taux_acces_ratio'] = taux_acces_ratio

    # Réinitialiser l'index pour avoir un dataframe propre
    df_final = df_final.reset_index(drop=True)

    print(f"DataFrame final créé avec {len(df_final)} lignes et {len(df_final.columns)} colonnes")
    print(f"\nColonnes incluses:")
    print(df_final.columns.tolist())

    df_final.to_csv("data/df_features.csv", index=False)
    return df_final


def clean_data(df_final, alpha_geographique=0.98):

    # Nettoyage des données géographiques
    print("\n" + "="*80)
    print("NETTOYAGE DES COORDONNÉES GPS")
    print("="*80)
    
    # Extraire longitude et latitude
    df_final["longitude"] = df_final["Coordonnées GPS de la formation"].str.split(",").str[0]
    df_final["latitude"] = df_final["Coordonnées GPS de la formation"].str.split(",").str[1]

    df_final["longitude"] = df_final["longitude"].astype(float)
    df_final["latitude"] = df_final["latitude"].astype(float)
    
    # Statistiques avant clipping
    print("\nStatistiques des coordonnées AVANT clipping:")
    print(f"  Longitude: min={df_final['longitude'].min():.4f}, max={df_final['longitude'].max():.4f}, "
          f"médiane={df_final['longitude'].median():.4f}")
    print(f"  Latitude: min={df_final['latitude'].min():.4f}, max={df_final['latitude'].max():.4f}, "
          f"médiane={df_final['latitude'].median():.4f}")
    
    # Clipping des valeurs extrêmes (Winsorization) - comprime les outliers
    # Utilise les percentiles 1 et 99 pour garder 98% des données intactes
    percentile_low = (1 - alpha_geographique) * 100  # 1st percentile
    percentile_high = alpha_geographique * 100  # 99th percentile
    
    lon_low = df_final["longitude"].quantile(percentile_low / 100)
    lon_high = df_final["longitude"].quantile(percentile_high / 100)
    lat_low = df_final["latitude"].quantile(percentile_low / 100)
    lat_high = df_final["latitude"].quantile(percentile_high / 100)
    
    # Compter les valeurs qui seront clippées
    n_outliers_lon = ((df_final["longitude"] < lon_low) | (df_final["longitude"] > lon_high)).sum()
    n_outliers_lat = ((df_final["latitude"] < lat_low) | (df_final["latitude"] > lat_high)).sum()
    
    print(f"\nClipping des valeurs extrêmes (percentiles {percentile_low}% - {percentile_high}%):")
    print(f"  Longitude: [{lon_low:.4f}, {lon_high:.4f}]")
    print(f"  Latitude: [{lat_low:.4f}, {lat_high:.4f}]")
    print(f"  Valeurs clippées: {n_outliers_lon} longitudes, {n_outliers_lat} latitudes")
    
    # Clipper les valeurs extrêmes
    df_final["longitude"] = df_final["longitude"].clip(lower=lon_low, upper=lon_high)
    df_final["latitude"] = df_final["latitude"].clip(lower=lat_low, upper=lat_high)
    
    # Statistiques après clipping
    print("\nStatistiques des coordonnées APRÈS clipping:")
    print(f"  Longitude: min={df_final['longitude'].min():.4f}, max={df_final['longitude'].max():.4f}")
    print(f"  Latitude: min={df_final['latitude'].min():.4f}, max={df_final['latitude'].max():.4f}")
    
    # Normalisation MinMax après clipping
    scaler = MinMaxScaler()
    df_final[["longitude", "latitude"]] = scaler.fit_transform(df_final[["longitude", "latitude"]])
    
    print("\nNormalisation MinMax appliquée (valeurs entre 0 et 1)")
    print(f"  Longitude normalisée: min={df_final['longitude'].min():.4f}, max={df_final['longitude'].max():.4f}")
    print(f"  Latitude normalisée: min={df_final['latitude'].min():.4f}, max={df_final['latitude'].max():.4f}")
    print("="*80)
    
    print("\n" + "="*80)
    print("NETTOYAGE DES DONNÉES AVANT NORMALISATION")
    print("="*80)

    # Identifier et gérer les valeurs problématiques (inf, -inf, NaN)
    numeric_cols_check = df_final.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nVérification des valeurs problématiques dans {len(numeric_cols_check)} colonnes numériques:")
    has_issues = False

    for col in numeric_cols_check:
        n_inf = np.isinf(df_final[col]).sum()
        n_nan = df_final[col].isna().sum()
        if n_inf > 0 or n_nan > 0:
            print(f"  ⚠️  {col}: {n_inf} inf, {n_nan} NaN")
            has_issues = True

    if not has_issues:
        print("  ✓ Aucune valeur problématique détectée")

    # Remplacer les valeurs infinies et NaN
    print("\nTraitement des valeurs problématiques:")
    print("  - inf → remplacé par 1")
    print("  - -inf → remplacé par -1")
    print("  - NaN → remplacé par la médiane de la colonne")

    df_final_cleaned = df_final.copy()

    for col in numeric_cols_check:
        # Remplacer inf par 1 et -inf par -1
        df_final_cleaned[col] = df_final_cleaned[col].replace([np.inf, -np.inf], [1, -1])
        
        # Remplacer NaN par la médiane (plus robuste que la moyenne pour les outliers)
        if df_final_cleaned[col].isna().sum() > 0:
            median_val = df_final_cleaned[col].median()
            # Si la médiane est aussi NaN (toute la colonne est NaN), utiliser 0
            if pd.isna(median_val):
                median_val = 0
            df_final_cleaned[col] = df_final_cleaned[col].fillna(median_val)
            print(f"  ✓ {col}: rempli avec médiane = {median_val:.4f}")

    # Vérification finale
    n_inf_total = np.isinf(df_final_cleaned[numeric_cols_check]).sum().sum()
    n_nan_total = df_final_cleaned[numeric_cols_check].isna().sum().sum()
    print(f"\n✓ Après nettoyage: {n_inf_total} inf, {n_nan_total} NaN")

    # Sauvegarder le dataframe nettoyé
    df_final_cleaned.to_csv("data/df_features_cleaned.csv", index=False)
    print(f"✓ DataFrame nettoyé sauvegardé dans 'data/df_features_cleaned.csv'")
    return df_final_cleaned

# =============================================================================
# NORMALISATION DES FEATURES POUR LE CLUSTERING
# =============================================================================

def normalize_features_for_clustering(df, exclude_cols=None, method='minmax_symmetric'):
    """
    Normalise les features numériques d'un dataframe pour le clustering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Le dataframe à normaliser
    exclude_cols : list, optional
        Liste des colonnes à exclure de la normalisation (par exemple les identifiants)
    method : str, default='minmax_symmetric'
        Méthode de normalisation:
        - 'minmax_symmetric': MinMaxScaler vers [-1, +1] - recommandé pour avoir toutes les colonnes sur la même échelle
        - 'standard': StandardScaler (z-score)
        - 'minmax': MinMaxScaler (0-1)
        - 'robust': RobustScaler (médiane et IQR, résistant aux outliers)
    
    Returns:
    --------
    df_normalized : pd.DataFrame
        Dataframe avec les colonnes numériques normalisées
    scaler : Scaler object ou None
        L'objet scaler utilisé (None pour minmax_symmetric car normalisation manuelle)
    numeric_cols : list
        Liste des colonnes numériques qui ont été normalisées
    """
    
    
    df_normalized = df.copy()
    
    # Identifier les colonnes numériques
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclure certaines colonnes si spécifié
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Choisir le scaler
    if method == 'minmax_symmetric':
        print("📊 Utilisation de MinMaxScaler symétrique ([-1, +1] normalization)")
        print("   Formule: 2 * (X - min) / (max - min) - 1")
        
        # Normalisation manuelle vers [-1, +1] pour chaque colonne
        for col in numeric_cols:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            
            if max_val - min_val != 0:
                # Normaliser vers [-1, +1]
                df_normalized[col] = 2 * (df_normalized[col] - min_val) / (max_val - min_val) - 1
            else:
                # Si la colonne est constante, mettre à 0
                df_normalized[col] = 0
        
        scaler = None  # Pas de scaler sklearn pour cette méthode custom
        
    elif method == 'standard':
        scaler = StandardScaler()
        print("📊 Utilisation de StandardScaler (z-score normalization)")
        df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
        
    elif method == 'minmax':
        scaler = MinMaxScaler()
        print("📊 Utilisation de MinMaxScaler (0-1 normalization)")
        df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
        
    elif method == 'robust':
        scaler = RobustScaler()
        print("📊 Utilisation de RobustScaler (robust to outliers)")
        df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    print(f"✓ {len(numeric_cols)} colonnes numériques normalisées")
    
    # Vérification des min/max pour minmax_symmetric
    if method == 'minmax_symmetric':
        print("\n📈 Vérification des valeurs normalisées:")
        for col in numeric_cols[:5]:  # Afficher les 5 premières colonnes
            print(f"   {col}: min={df_normalized[col].min():.4f}, max={df_normalized[col].max():.4f}")
        if len(numeric_cols) > 5:
            print(f"   ... ({len(numeric_cols) - 5} autres colonnes)")
    
    return df_normalized, scaler, numeric_cols


def visualize_normalization_impact(df_original, df_normalized, numeric_cols, save_prefix='data/normalization'):
    """
    Visualise l'impact de la normalisation sur les features.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Dataframe avant normalisation
    df_normalized : pd.DataFrame
        Dataframe après normalisation
    numeric_cols : list
        Liste des colonnes numériques normalisées
    save_prefix : str
        Préfixe pour sauvegarder les graphiques
    """
    
    # Filtrer uniquement les colonnes *_ratio pour la visualisation
    ratio_cols = [col for col in numeric_cols if col.endswith('_ratio')]
    
    print(f"📊 Visualisation de {len(ratio_cols)} features *_ratio (sur {len(numeric_cols)} features totales)")
    print(f"   Features visualisées: {', '.join(ratio_cols)}")
    
    n_features = len(ratio_cols)
    
    # =========================================================================
    # VISUALISATION 1: Comparaison avant/après pour quelques features
    # =========================================================================
    n_samples = min(8, n_features)
    sample_features = ratio_cols[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(sample_features):
        # Avant normalisation
        axes[i, 0].hist(df_original[col].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[i, 0].set_title(f'AVANT: {col}', fontsize=11, fontweight='bold')
        axes[i, 0].set_xlabel('Valeur', fontsize=9)
        axes[i, 0].set_ylabel('Fréquence', fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)
        
        # Après normalisation
        axes[i, 1].hist(df_normalized[col].dropna(), bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[i, 1].set_title(f'APRÈS: {col}', fontsize=11, fontweight='bold')
        axes[i, 1].set_xlabel('Valeur normalisée', fontsize=9)
        axes[i, 1].set_ylabel('Fréquence', fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Comparaison des distributions AVANT et APRÈS normalisation (échantillon)', 
                 fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_sample_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique '{save_prefix}_sample_comparison.png' sauvegardé")
    plt.show()
    
    # =========================================================================
    # VISUALISATION 2: Box plots comparatifs pour toutes les features *_ratio
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(10, n_features * 0.35)))
    
    # Avant normalisation
    positions = np.arange(len(ratio_cols))
    bp1 = ax1.boxplot([df_original[col].dropna() for col in ratio_cols], 
                       vert=False, patch_artist=True, positions=positions, widths=0.6)
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(ratio_cols)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_yticks(positions)
    ax1.set_yticklabels(ratio_cols, fontsize=8)
    ax1.set_xlabel('Valeur originale', fontsize=10, fontweight='bold')
    ax1.set_title('AVANT normalisation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Après normalisation
    bp2 = ax2.boxplot([df_normalized[col].dropna() for col in ratio_cols], 
                       vert=False, patch_artist=True, positions=positions, widths=0.6)
    
    colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(ratio_cols)))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_yticks(positions)
    ax2.set_yticklabels(ratio_cols, fontsize=8)
    ax2.set_xlabel('Valeur normalisée [-1, +1]', fontsize=10, fontweight='bold')
    ax2.set_title('APRÈS normalisation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=-1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle('Distribution des features *_ratio AVANT et APRÈS normalisation [-1, +1]', 
                 fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_all_boxplots.png', dpi=300, bbox_inches='tight')
    print(f"✓ Graphique '{save_prefix}_all_boxplots.png' sauvegardé")
    plt.show()
    
    # =========================================================================
    # RÉSUMÉ STATISTIQUE
    # =========================================================================
    print("\n" + "="*80)
    print("RÉSUMÉ DE LA NORMALISATION")
    print("="*80)
    
    print(f"\nNombre de features normalisées: {len(numeric_cols)}")
    print(f"  dont {len(ratio_cols)} features *_ratio visualisées")
    
    print(f"\nStatistiques des features *_ratio AVANT normalisation:")
    for col in ratio_cols:
        print(f"  {col}: min={df_original[col].min():.4f}, max={df_original[col].max():.4f}, mean={df_original[col].mean():.4f}")
    
    print(f"\nStatistiques des features *_ratio APRÈS normalisation:")
    for col in ratio_cols:
        print(f"  {col}: min={df_normalized[col].min():.4f}, max={df_normalized[col].max():.4f}, mean={df_normalized[col].mean():.4f}")
    
    print("\n✓ Toutes les colonnes sont maintenant sur l'échelle [-1, +1]")
    print("="*80)



def prepare_categorical_features_for_clustering(df):
    """
    Encode les features catégorielles pour le clustering.
    
    Stratégie:
    - Supprime: Code UAI, Établissement, Coordonnées GPS, Commune
    - One-hot encode (normalisé vers [-1, +1]): Statut, Filière, Région, Département
    
    Returns: df_encoded, n_new_features
    """
    df_encoded = df.copy()
    
    # Colonnes à supprimer (identifiants et haute cardinalité)
    cols_to_remove = [
        "Code UAI de l'établissement",
        "Établissement",
        "Coordonnées GPS de la formation",
        "Commune de l'établissement"
    ]
    
    for col in cols_to_remove:
        if col in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=[col])
    
    # Colonnes à one-hot encoder
    cols_to_encode = [
        "Statut de l'établissement de la filière de formation (public, privé…)",
        "Filière de formation",
        "Filière de formation très agrégée",
        "Filière de formation détaillée bis",
        "Département de l'établissement",
        "Région de l'établissement"
    ]
    
    n_new_features = 0
    for col in cols_to_encode:
        if col in df_encoded.columns:
            # One-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col[:25], drop_first=False)
            # Normaliser vers [-1, 1]: 0 -> -1, 1 -> 1
            dummies = dummies * 2 - 1
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            n_new_features += len(dummies.columns)
    
    # Supprimer toute colonne catégorielle restante
    remaining_cat = df_encoded.select_dtypes(exclude=['number']).columns.tolist()
    if remaining_cat:
        df_encoded = df_encoded.drop(columns=remaining_cat)
    
    return df_encoded, n_new_features



# =============================================================================
# CODE PRINCIPAL - Ne s'exécute que si ce fichier est lancé directement
# =============================================================================
if __name__ == "__main__":
    df_final = load_raw_data()


    # =============================================================================
    # APPLICATION DE LA NORMALISATION ET DU CLEANING
    # =============================================================================

    # Exclure les colonnes catégorielles et garder uniquement les numériques
    categorical_cols = ["Code UAI de l'établissement", 
                       "Établissement",
                       "Statut de l'établissement de la filière de formation (public, privé…)",
                       "Département de l'établissement",
                       "Région de l'établissement",
                       "Commune de l'établissement",
                       "Coordonnées GPS de la formation",
                       "Filière de formation",
                       "Filière de formation très agrégée",
                       "Filière de formation détaillée bis"]



    print("\n" + "="*80)
    print("NORMALISATION DES FEATURES POUR LE CLUSTERING")
    print("="*80)


    df_final_cleaned = clean_data(df_final)
    # Appliquer la normalisation MinMax symétrique [-1, +1] sur chaque colonne
    df_normalized, scaler, numeric_cols_normalized = normalize_features_for_clustering(
        df_final_cleaned, 
        exclude_cols=categorical_cols,
        method='robust'
    )

    # Sauvegarder le dataframe normalisé
    df_normalized.to_csv("data/df_features_normalized.csv", index=False)
    print(f"✓ DataFrame normalisé sauvegardé dans 'data/df_features_normalized.csv'")

    # # Visualiser l'impact de la normalisation

    # visualize_normalization_impact(df_final_cleaned, df_normalized, numeric_cols_normalized)


    # =============================================================================
    # INTÉGRATION DES FEATURES CATÉGORIELLES POUR LE CLUSTERING
    # =============================================================================


    # =============================================================================
    # APPLICATION DE L'ENCODAGE CATÉGORIEL
    # =============================================================================

    # print("\n" + "="*80)
    # print("ENCODAGE DES FEATURES CATÉGORIELLES")
    # print("="*80)

    # df_clustering, n_new = prepare_categorical_features_for_clustering(df_normalized)

    # print(f"\n✓ Encodage terminé:")
    # print(f"  - {n_new} nouvelles features créées")
    # print(f"  - {len(df_clustering.columns)} features totales")
    # print(f"  - {len(df_clustering)} lignes")

    # # Sauvegarder
    # df_clustering.to_csv("data/df_fetures_normalized_categorical.csv", index=False)



