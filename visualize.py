import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Define features to include (from main.py)
CATEGORICAL_FEATURES = [
    # Excluded: "Département de l'établissement"
    # Excluded: "Filière de formation très agrégée"
    # Excluded: "Filière de formation détaillée bis"
    "Statut de l'établissement de la filière de formation (public, privé…)"
]

NUMERICAL_FEATURES = [
    'Rang du dernier appelé du groupe 1',
    "Capacité de l'établissement par formation",
    'Effectif total des candidats en phase principale',
    'f_ratio_candidats',
    'f_ratio_admis',
    'f_selectivity_candidats',
    'b_ratio_candidats',
    'b_ratio_admis',
    'b_selectivity_candidats',
    'gen_candidats_ratio',
    'tech_candidats_ratio',
    'prof_candidats_ratio',
    'gen_admis_ratio',
    'tech_admis_ratio',
    'prof_admis_ratio',
    'gen_selectivity_candidats',
    'tech_selectivity_candidats',
    'prof_selectivity_candidats',
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


def plot_correlation_matrix(csv_path='data/df_features_normalized.csv'):
    """
    Charge les données normalisées et trace la matrice de corrélation
    des features spécifiées.
    """
    # Charger les données
    df = pd.read_csv(csv_path)
    
    # Sélectionner les colonnes qui existent dans le DataFrame
    all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    available_features = [col for col in all_features if col in df.columns]
    
    print(f"Features disponibles: {len(available_features)}/{len(all_features)}")
    missing_features = [col for col in all_features if col not in df.columns]
    if missing_features:
        print(f"Features manquantes: {missing_features}")
    
    # Créer un DataFrame avec les features sélectionnées
    df_selected = df[available_features].copy()
    
    # Encoder les variables catégorielles
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df_selected.columns:
            le = LabelEncoder()
            df_selected[cat_col] = le.fit_transform(df_selected[cat_col].astype(str))
    
    # Calculer la matrice de corrélation
    corr_matrix = df_selected.corr()
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(20, 18))
    
    # Créer le heatmap
    sns.heatmap(
        corr_matrix,
        annot=False,  # Ne pas afficher les valeurs pour plus de clarté
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Corrélation"},
        ax=ax
    )
    
    # Configurer les labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Titre
    plt.title('Matrice de Corrélation des Features (Données Normalisées)', 
              fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = 'data/correlation_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Matrice de corrélation sauvegardée: {output_path}")
    
    plt.show()
    
    # Afficher les corrélations les plus fortes (en valeur absolue)
    print("\n" + "="*80)
    print("Top 20 des corrélations les plus fortes (|r| > 0.5):")
    print("="*80)
    
    # Extraire le triangle supérieur de la matrice de corrélation
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_pairs = corr_matrix.where(mask).stack().sort_values(key=abs, ascending=False)
    
    # Filtrer les corrélations > 0.5 en valeur absolue
    strong_corr = corr_pairs[abs(corr_pairs) > 0.5].head(20)
    
    for (var1, var2), corr_value in strong_corr.items():
        print(f"{var1:40s} <-> {var2:40s} : {corr_value:6.3f}")
    
    return corr_matrix


def plot_normalized_distributions(csv_path='data/df_features_normalized.csv'):
    """
    Visualise les distributions des features normalisées pour vérifier
    la qualité de la normalisation.
    """
    # Charger les données
    df = pd.read_csv(csv_path)
    
    # Features à exclure des visualisations (distributions trop extrêmes)
    EXCLUDED_FEATURES = ['tech_candidats_ratio', 'prof_candidats_ratio']
    
    # Sélectionner uniquement les features numériques disponibles (sauf celles exclues)
    available_numerical = [col for col in NUMERICAL_FEATURES 
                          if col in df.columns and col not in EXCLUDED_FEATURES]
    
    print(f"\n{'='*80}")
    print(f"ANALYSE DES DISTRIBUTIONS - Features disponibles: {len(available_numerical)}/{len(NUMERICAL_FEATURES)}")
    print(f"Features exclues (distributions extrêmes): {EXCLUDED_FEATURES}")
    print(f"{'='*80}")
    
    # Créer un DataFrame avec les features numériques
    df_numeric = df[available_numerical].copy()
    
    # Calculer les statistiques descriptives
    stats = df_numeric.describe().T
    stats['std_dev'] = df_numeric.std()
    stats['variance'] = df_numeric.var()
    stats['skewness'] = df_numeric.skew()
    stats['kurtosis'] = df_numeric.kurtosis()
    
    print("\nStatistiques descriptives des features normalisées:")
    print("-" * 80)
    print(stats[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].to_string())
    
    # Vérifier si les données sont normalisées (moyenne ~0, std ~1)
    print("\n" + "="*80)
    print("VÉRIFICATION DE LA NORMALISATION:")
    print("="*80)
    mean_check = stats['mean'].abs().mean()
    std_check = stats['std'].mean()
    print(f"Moyenne absolue des moyennes: {mean_check:.4f} (devrait être proche de 0)")
    print(f"Moyenne des écarts-types: {std_check:.4f} (devrait être proche de 1 pour standard, variable pour robust)")
    
    # === PLOT 1: Boxplot de toutes les features ===
    fig1, ax1 = plt.subplots(figsize=(20, 10))
    
    # Préparer les données pour le boxplot
    df_melted = df_numeric.melt(var_name='Feature', value_name='Valeur Normalisée')
    
    # Créer le boxplot
    sns.boxplot(
        data=df_melted,
        x='Feature',
        y='Valeur Normalisée',
        palette='Set2',
        ax=ax1
    )
    
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Valeur Normalisée', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution des Features Normalisées (Boxplot)', 
                  fontsize=16, pad=20, fontweight='bold')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Moyenne théorique (0)')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    plt.tight_layout()
    output_path1 = 'data/normalized_distributions_boxplot.png'
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"\nBoxplot sauvegardé: {output_path1}")
    plt.close()
    
    # === PLOT 2: Violin plots pour un échantillon de features ===
    # Sélectionner un sous-ensemble de features pour la lisibilité
    sample_features = available_numerical[:min(15, len(available_numerical))]
    
    fig2, ax2 = plt.subplots(figsize=(20, 10))
    
    df_sample = df[sample_features].melt(var_name='Feature', value_name='Valeur Normalisée')
    
    sns.violinplot(
        data=df_sample,
        x='Feature',
        y='Valeur Normalisée',
        palette='muted',
        ax=ax2
    )
    
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Valeur Normalisée', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution des Features Normalisées (Violin Plot - Échantillon)', 
                  fontsize=16, pad=20, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Moyenne théorique (0)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path2 = 'data/normalized_distributions_violin.png'
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Violin plot sauvegardé: {output_path2}")
    plt.close()
    
    # === PLOT 3: Histogrammes de quelques features représentatives ===
    n_features_hist = min(12, len(available_numerical))
    n_cols = 4
    n_rows = (n_features_hist + n_cols - 1) // n_cols
    
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, feature in enumerate(available_numerical[:n_features_hist]):
        ax = axes[idx]
        
        # Histogramme
        ax.hist(df_numeric[feature].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Statistiques
        mean_val = df_numeric[feature].mean()
        std_val = df_numeric[feature].std()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'μ={mean_val:.3f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'σ={std_val:.3f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_title(feature, fontsize=9, fontweight='bold')
        ax.set_xlabel('Valeur', fontsize=8)
        ax.set_ylabel('Fréquence', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    # Cacher les axes inutilisés
    for idx in range(n_features_hist, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Histogrammes des Features Normalisées', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path3 = 'data/normalized_distributions_histograms.png'
    plt.savefig(output_path3, dpi=300, bbox_inches='tight')
    print(f"Histogrammes sauvegardés: {output_path3}")
    plt.close()
    
    # === PLOT 4: Heatmap des statistiques par feature ===
    fig4, ax4 = plt.subplots(figsize=(12, max(10, len(available_numerical) * 0.3)))
    
    stats_display = stats[['mean', 'std', 'min', 'max', 'skewness']].T
    
    sns.heatmap(
        stats_display,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn_r',
        center=0,
        cbar_kws={"label": "Valeur"},
        ax=ax4
    )
    
    ax4.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Statistiques', fontsize=12, fontweight='bold')
    ax4.set_title('Statistiques des Features Normalisées (Heatmap)', 
                  fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    output_path4 = 'data/normalized_statistics_heatmap.png'
    plt.savefig(output_path4, dpi=300, bbox_inches='tight')
    print(f"Heatmap des statistiques sauvegardée: {output_path4}")
    plt.close()
    
    print("\n" + "="*80)
    print("VISUALISATIONS TERMINÉES")
    print("="*80)
    
    return stats


def plot_pca_simple(X_pca, labels=None, pca=None, output_path='data/clustering_vis/pca_plot.png', 
                   title='PCA Visualization', colorbar_label='Cluster', cmap='viridis', 
                   categorical=False):
    """
    Simple function to plot PCA results.
    
    Parameters:
    -----------
    X_pca : array-like, shape (n_samples, 2)
        The 2D PCA-transformed data
    labels : array-like, optional
        Values for coloring points (cluster labels, feature values, etc.)
        Can be numerical or categorical (strings)
    pca : PCA object, optional
        Fitted PCA object to get explained variance ratios
    output_path : str
        Path to save the figure
    title : str
        Plot title
    colorbar_label : str
        Label for the colorbar/legend
    cmap : str
        Colormap name
    categorical : bool
        If True, treat labels as categorical and use discrete colors with a legend.
        If False, treat as continuous and use a colorbar.
        If None (default), auto-detect based on label dtype.
    """
    import os
    from matplotlib.colors import ListedColormap
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot with or without labels
    if labels is not None:
        labels_array = np.array(labels)
        
        # Auto-detect categorical if not specified
        if categorical is None:
            categorical = labels_array.dtype == 'object' or labels_array.dtype.name == 'category'
        
        if categorical:
            # Handle categorical variables
            unique_labels = pd.unique(labels_array)
            n_categories = len(unique_labels)
            
            # Choose simple, vivid, and distinct colors
            if n_categories == 2:
                # Strong red and blue
                colors = np.array([[0.1216, 0.4667, 0.7059, 1.0],  # blue (#1f77b4)
                                   [0.8392, 0.1529, 0.1569, 1.0]]) # red (#d62728)
            elif n_categories <= 10:
                # Bright qualitative palette
                import seaborn as sns
                colors = np.array(sns.color_palette('bright', n_categories))
            elif n_categories <= 20:
                # Use tab20 which is high-contrast
                colors = plt.cm.tab20(np.linspace(0, 1, n_categories))
            elif n_categories <= 32:
                # Combine tab20 and tab20b for more distinct colors
                base1 = plt.cm.tab20(np.linspace(0, 1, 20))
                base2 = plt.cm.tab20b(np.linspace(0, 1, 12))
                colors = np.vstack([base1, base2])[:n_categories]
            else:
                # Many categories: evenly spaced hues, high saturation
                import seaborn as sns
                colors = np.array(sns.hls_palette(n_categories, l=0.5, s=0.9))
            
            # Create a mapping from categories to colors
            category_to_color = {cat: colors[i] for i, cat in enumerate(unique_labels)}
            point_colors = [category_to_color[label] for label in labels_array]
            
            # Plot
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=point_colors, 
                               alpha=0.6, edgecolors='k', s=50)
            
            # Create legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=category_to_color[cat], 
                                    edgecolor='k', 
                                    label=str(cat)) 
                              for cat in unique_labels]
            
            # Position legend outside plot if many categories
            if n_categories > 6:
                ax.legend(handles=legend_elements, 
                         title=colorbar_label,
                         bbox_to_anchor=(1.05, 1), 
                         loc='upper left',
                         fontsize=9,
                         title_fontsize=10)
            else:
                ax.legend(handles=legend_elements, 
                         title=colorbar_label,
                         loc='best',
                         fontsize=9,
                         title_fontsize=10)
        else:
            # Handle continuous variables (numeric)
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                               c=labels_array, cmap=cmap, 
                               alpha=0.6, edgecolors='k', s=50)
            cbar = plt.colorbar(scatter, ax=ax, label=colorbar_label)
            cbar.ax.tick_params(labelsize=10)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                  alpha=0.6, edgecolors='k', s=50, color='steelblue')
    
    # Set labels
    if pca is not None:
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                     fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA plot saved to: {output_path}")
    plt.close()
    
    return fig


if __name__ == "__main__":
    print("=" * 80)
    print("GÉNÉRATION DE LA MATRICE DE CORRÉLATION")
    print("=" * 80)
    correlation_matrix = plot_correlation_matrix()
    
    print("\n\n")
    print("=" * 80)
    print("ANALYSE DES DISTRIBUTIONS NORMALISÉES")
    print("=" * 80)
    stats = plot_normalized_distributions()

