from features import load_raw_data, clean_data, normalize_features_for_clustering
from clustering import load_and_prepare_data, perform_clustering, print_metrics, perform_pca, create_interactive_plot, create_static_visualizations
from clustering import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from visualize import plot_pca_simple
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

categorical_cols = ["Code UAI de l'établissement", 
                   "Établissement",
                   "Statut de l’établissement de la filière de formation (public, privé…)",
                   "Département de l'établissement",
                   "Région de l'établissement",
                   "Commune de l'établissement",
                   "Coordonnées GPS de la formation",
                   "Filière de formation",
                   "Filière de formation.1",
                   "Filière de formation très agrégée",
                   "Filière de formation détaillée bis"]

CATEGORICAL_FEATURES = [
    # "Département de l’établissement",
    # "Filière de formation très agrégée",
    "Filière de formation détaillée bis",
    "Statut de l’établissement de la filière de formation (public, privé…)"
]

NUMERICAL_FEATURES = [
    'Rang du dernier appelé du groupe 1',
    'Capacité de l’établissement par formation',
    'Effectif total des candidats en phase principale',
    'f_ratio_candidats',
    'f_ratio_admis',
    'f_selectivity_candidats',
    'b_ratio_candidats',
    'b_ratio_admis',
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


if __name__ == "__main__":
    # filieres1 = ["Ecole d'Ingénieur", "CPGE", "BTS", "BUT", "Licence", "Licence_Las", "IFSI", "PASS", "EFTS", "Ecole de Commerce", "Autre Formation"]
    # filieres2 = [None for _ in range(len(filieres1))]

    filieres1 = ["CPGE"]
    filieres2= [None]

    normalize_method = 'robust'
    n_clusters = 3


    ## NORMALISATION DES FEATURES
    df_final = load_raw_data("data/fr-esr-parcoursup.csv", filieres1, filieres2)

    # alpha_geographique donne la valmeur que l'on donne au données géographiques (1.00 : les données géo sont très représentées, <0.9 : les données géo sont très peu représentées)
    df_final_cleaned = clean_data(df_final, alpha_geographique=0.965)

    df_normalized, scaler, numeric_cols_normalized = normalize_features_for_clustering(
        df_final_cleaned, 
        exclude_cols=categorical_cols,
        method=normalize_method
    )

    df_normalized.to_csv("data/df_features_normalized.csv", index=False)


    ## CLUSTERING
    df_clustering, X = load_and_prepare_data("data/df_features_normalized.csv", NUMERICAL_FEATURES, CATEGORICAL_FEATURES)

    # Metrics lists
    sil_scores = []  # Silhouette (exclude noise)
    ch_scores = []   # Calinski-Harabasz
    db_scores = []   # Davies-Bouldin
    dbscan_params = []  # (eps, min_samples, n_clusters_effective)
    # Perform clustering
    # kmeans, labels_kmeans = perform_clustering(X, n_clusters=n_clusters)
    for eps in [3.0]:
        for min_samples in [6]:
            try:
                dbscan, labels_dbscan = perform_clustering(X, n_clusters=n_clusters, method='dbscan', eps=eps, min_samples=min_samples)

                # df_clustering_kmeans = df_clustering.copy()
                df_clustering_dbscan = df_clustering.copy()

                # df_clustering_kmeans['cluster'] = labels_kmeans
                df_clustering_dbscan['cluster'] = labels_dbscan
                n_clusters_dbscan = len(np.unique(labels_dbscan))
                print(f"DBSCAN with eps={eps} and min_samples={min_samples} has {n_clusters_dbscan} clusters")

                # Compute CH and DB safely (require >=2 clusters; exclude noise -1)
                labels_for_metric = labels_dbscan
                if -1 in labels_for_metric:
                    mask = labels_for_metric != -1
                    labels_for_metric = labels_for_metric[mask]
                    X_for_metric = X[mask]
                else:
                    X_for_metric = X

                if len(np.unique(labels_for_metric)) >= 2:
                    try:
                        sil_val = silhouette_score(X_for_metric, labels_for_metric)
                    except Exception:
                        sil_val = None
                    try:
                        ch_val = calinski_harabasz_score(X_for_metric, labels_for_metric)
                    except Exception:
                        ch_val = None
                    try:
                        db_val = davies_bouldin_score(X_for_metric, labels_for_metric)
                    except Exception:
                        db_val = None
                else:
                    sil_val = None
                    ch_val = None
                    db_val = None

                sil_scores.append(sil_val)
                ch_scores.append(ch_val)
                db_scores.append(db_val)
                dbscan_params.append((eps, min_samples, len(np.unique(labels_for_metric))))

                # Quick feedback
                sil_txt = f"{sil_val:.3f}" if sil_val is not None else "N/A"
                ch_txt = f"{ch_val:.1f}" if ch_val is not None else "N/A"
                db_txt = f"{db_val:.3f}" if db_val is not None else "N/A"
                print(f"  Sil={sil_txt}  CH={ch_txt}  DB={db_txt}")
                # Save results
                # df_clustering_kmeans.to_csv('data/df_clustering_kmeans.csv', index=False)
                df_clustering_dbscan.to_csv('data/df_clustering_dbscan.csv', index=False)

                # Print metrics
                # print_metrics(X, labels_kmeans, kmeans)
                
                # Perform PCA
                pca, X_pca = perform_pca(X)
                
                # Load cleaned data with all columns for plotting
                df_for_plotting = pd.read_csv('data/df_features_cleaned.csv')
                df_for_plotting['cluster'] = labels_dbscan

                # Create visualizations
                # create_interactive_plsot(df_clustering_dbscan, X_pca, pca)
                silhouette_score = create_static_visualizations(df_clustering_dbscan, X, X_pca, pca, dbscan, labels_dbscan, output_path=f'data/clustering_vis/dbscan_eps_{eps}_min_samples_{min_samples}.png', output_dir=f'data/clustering_vis/dbscan_eps_{eps}_min_samples_{min_samples}')
                
                # silhouette_scores.append(silhouette_score)
            except Exception as e:
                print(f"Warning: {e}")
                continue


    # Report top DBSCAN configs by silhouette
    results = []
    for (eps, ms, k_eff), sil, ch, db in zip(dbscan_params, sil_scores, ch_scores, db_scores):
        if sil is not None:
            results.append((sil, eps, ms, k_eff, ch, db))
    results.sort(reverse=True, key=lambda x: x[0])

    print("\nTop DBSCAN configs by silhouette (excluding noise):")
    for row in results[:5]:
        sil, eps, ms, k_eff, ch, db = row
        ch_txt = f"{ch:.1f}" if ch is not None else "N/A"
        db_txt = f"{db:.3f}" if db is not None else "N/A"
        print(f"  eps={eps}, min_samples={ms}, clusters={k_eff} -> Sil={sil:.3f}, CH={ch_txt}, DB={db_txt}")

    # Create PCA plots for top 3 configs
    if results:
        pca, X_pca = perform_pca(X)
        for row in results[:3]:
            sil, eps, ms, k_eff, ch, db = row
            _, labels_best = perform_clustering(X, method='dbscan', eps=eps, min_samples=ms, n_clusters=0)
            title = f"DBSCAN eps={eps}, min_samples={ms} | Sil={sil:.3f}"
            out = f"data/clustering_vis/pca_dbscan_eps_{eps}_min_{ms}.png"
            plot_pca_simple(X_pca, labels=labels_best, pca=pca, output_path=out, title=title, categorical=True)

    # Plot Silhouette heatmap over (eps, min_samples)
    if dbscan_params:
        import os
        os.makedirs('data/clustering_vis', exist_ok=True)
        eps_list = sorted(list({p[0] for p in dbscan_params}))
        ms_list = sorted(list({p[1] for p in dbscan_params}))
        grid = np.full((len(ms_list), len(eps_list)), np.nan, dtype=float)
        # fill grid
        for (eps, ms, _), sil in zip(dbscan_params, sil_scores):
            i = ms_list.index(ms)
            j = eps_list.index(eps)
            if sil is not None:
                grid[i, j] = sil

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(grid, origin='lower', cmap='viridis', aspect='auto',
                       vmin=np.nanmin(grid), vmax=np.nanmax(grid))
        ax.set_xticks(range(len(eps_list)))
        ax.set_xticklabels(eps_list)
        ax.set_yticks(range(len(ms_list)))
        ax.set_yticklabels(ms_list)
        ax.set_xlabel('eps')
        ax.set_ylabel('min_samples')
        ax.set_title('Silhouette score (excluding noise)')
        # annotate
        for i in range(len(ms_list)):
            for j in range(len(eps_list)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha='center', va='center', color='white', fontsize=8,
                            path_effects=[])
        cbar = fig.colorbar(im, ax=ax, label='Silhouette')
        fig.tight_layout()
        out_heat = 'data/clustering_vis/silhouette_dbscan_heatmap.png'
        fig.savefig(out_heat, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Silhouette heatmap saved to: {out_heat}")

    # Optional: plot using the second duplicate column "Filière de formation" from original CSV (column 14)
    # try:
    #     df_orig = pd.read_csv('data/fr-esr-parcoursup.csv', sep=';')
    #     # Filter like load_raw_data: keep CPGE only (high-level track)
    #     if 'Filière de formation très agrégée' in df_orig.columns:
    #         df_orig_filtered = df_orig[df_orig['Filière de formation très agrégée'] == 'CPGE'].copy()
    #     else:
    #         df_orig_filtered = df_orig.copy()

    #     # Use the duplicate column if present, otherwise fallback to positional 14th (1-based) column
    #     if 'Filière de formation.1' in df_orig_filtered.columns:
    #         labels_fil_col14 = df_orig_filtered['Filière de formation.1'].reset_index(drop=True)
    #     else:
    #         # Positional fallback: 14th column in 1-based indexing => index 13
    #         labels_fil_col14 = df_orig_filtered.iloc[:, 13].reset_index(drop=True)

    #     # Ensure alignment with X_pca rows
    #     if len(labels_fil_col14) == len(X_pca):
    #         plot_pca_simple(
    #             X_pca,
    #             labels=labels_fil_col14,
    #             pca=pca,
    #             output_path='data/clustering_vis/pca_Filière de CPGE.png',
    #             title='PCA - Filière de CPGE',
    #             colorbar_label='Filière de CPGE',
    #             categorical=True
    #         )
    # except Exception as e:
    #     print(f"Warning: could not plot 'Filière de formation' column 14 from original CSV: {e}")

    # # Plot simple PCA visualization for numerical features (using original non-normalized data for visualization)
    # # Note: Use curved apostrophes (') as they appear in the CSV
    # for carac in ["Capacité de l\u2019établissement par formation", "Effectif total des candidats en phase principale", "longitude", "latitude", 'taux_acces_ratio']:
    #     plot_pca_simple(X_pca, 
    #                 labels=df_for_plotting[carac], 
    #                 pca=pca, 
    #                 output_path=f'data/clustering_vis/pca_{carac}.png',
    #                 title=f'PCA - {carac}',
    #                 colorbar_label=carac,
    #                 cmap='RdYlGn',
    #                 categorical=False)
    
    # # Plot simple PCA visualization for categorical features
    # for carac in ["Filière de formation", "Filière de formation détaillée bis", "Statut de l\u2019établissement de la filière de formation (public, privé…)"]:
    #     plot_pca_simple(X_pca, 
    #                 labels=df_for_plotting[carac], 
    #                 pca=pca, 
    #                 output_path=f'data/clustering_vis/pca_{carac}.png',
    #                 title=f'PCA - {carac}',
    #                 colorbar_label=carac,
    #                 categorical=True)