import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATASETS = [
    ("data/df_clustering_kmeans_3.csv", "kmeans_3"),
    ("data/df_clustering_dbscan_3.5_6.csv", "dbscan_3.5_6"),
    ("data/df_hierarchical_clusters_3.csv", "hierarchical_3"),
]


def find_cluster_col(df):
    # Prefer the known names (French 'Groupe' for hierarchical), then common names.
    candidates = ["Groupe", "Group", "group", "cluster", "Cluster", "groupes"]
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lowmap = {col.lower(): col for col in df.columns}
    for want in ["groupe", "group", "cluster", "groupes"]:
        if want in lowmap:
            return lowmap[want]
    # final fallback: last column
    return df.columns[-1]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_numeric_features(df, cluster_col):
    features = [c for c in df.columns if c != cluster_col]
    numeric = []
    for f in features:
        ser = pd.to_numeric(df[f], errors='coerce')
        # keep feature if at least one non-NA numeric value exists
        if ser.notna().any():
            numeric.append(f)
    return numeric


def plot_boxplot(df, cluster_col, feature, outpath, dataset_name):
    # Coerce feature to numeric
    ser = pd.to_numeric(df[feature], errors='coerce')
    df_plot = df[[cluster_col]].copy()
    df_plot[feature] = ser

    groups = []
    labels = []
    # sort groups for deterministic order
    unique_groups = sorted(df_plot[cluster_col].dropna().unique(), key=lambda x: (str(x)))
    for g in unique_groups:
        arr = df_plot.loc[df_plot[cluster_col] == g, feature].dropna().values
        groups.append(arr)
        labels.append(str(g))

    if not groups:
        raise ValueError(f"No data to plot for feature {feature}")

    plt.figure(figsize=(6, 4))
    bplot = plt.boxplot(groups, labels=labels, patch_artist=True, showfliers=False)

    # style boxes
    for patch in bplot['boxes']:
        patch.set(facecolor='#8FBCE6', edgecolor='black', alpha=0.85)

    # calculate and plot means
    means = [np.mean(g) if len(g) > 0 else np.nan for g in groups]
    x = np.arange(1, len(means) + 1)
    plt.scatter(x, means, color='red', marker='D', label='mean')
    plt.legend()

    plt.xlabel('Group')
    plt.ylabel(feature)
    plt.title(f"{dataset_name} — {feature} (boxplot with mean)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def get_feature_importance(method, df, numeric_features, cluster_col):
    """Calculate feature importance ranking for each clustering method."""
    if method == "kmeans":
        # Pour k-means : différence entre les centres des clusters
        clusters = df[cluster_col].unique()
        if len(clusters) < 2:
            return {}
            
        importances = {}
        for feat in numeric_features:
            # Calculer les centres des clusters
            centers = [df[df[cluster_col] == c][feat].mean() for c in clusters]
            # La différence absolue maximale entre deux centres
            importance = max(abs(centers[i] - centers[j]) 
                           for i in range(len(centers)) 
                           for j in range(i + 1, len(centers)))
            importances[feat] = importance
            
    elif method == "dbscan":
        # Pour DBSCAN : variance entre les moyennes des clusters
        clusters = df[df[cluster_col] != -1][cluster_col].unique()
        if len(clusters) < 2:
            return {}
            
        importances = {}
        for feat in numeric_features:
            # Calculer la moyenne par cluster (excluant le bruit -1)
            cluster_means = []
            for c in clusters:
                mean = df[df[cluster_col] == c][feat].mean()
                cluster_means.append(mean)
            # Variance entre les moyennes
            importances[feat] = np.var(cluster_means)
            
    elif method == "hierarchical":
        # Pour hierarchical : même approche que k-means
        clusters = df[cluster_col].unique()
        if len(clusters) < 2:
            return {}
            
        importances = {}
        for feat in numeric_features:
            centers = [df[df[cluster_col] == c][feat].mean() for c in clusters]
            importance = max(abs(centers[i] - centers[j]) 
                           for i in range(len(centers)) 
                           for j in range(i + 1, len(centers)))
            importances[feat] = importance
    else:
        return {}
        
    # Convertir en rangs (1 = plus important)
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return {feat: rank+1 for rank, (feat, _) in enumerate(sorted_features)}

def main():
    base_out = "visualization_images"
    ensure_dir(base_out)

    created = []

    for path, shortname in DATASETS:
        if not os.path.exists(path):
            print(f"WARN: file not found: {path}")
            continue

        print(f"Processing {path}...")
        df = pd.read_csv(path)
        cluster_col = find_cluster_col(df)

        numeric_features = get_numeric_features(df, cluster_col)
        if not numeric_features:
            print(f"No numeric features found for {path}")
            continue

        # Déterminer la méthode de clustering à partir du shortname
        if "kmeans" in shortname:
            method = "kmeans"
        elif "dbscan" in shortname:
            method = "dbscan"
        elif "hierarchical" in shortname:
            method = "hierarchical"
        else:
            method = None
            
        # Calculer l'importance des features
        feature_ranks = get_feature_importance(method, df, numeric_features, cluster_col)

        outdir = os.path.join(base_out, shortname)
        ensure_dir(outdir)

        for feat in numeric_features:
            rank = feature_ranks.get(feat, len(numeric_features))  # default au dernier rang si pas de rang
            outfile = os.path.join(outdir, f"{rank:02d}_{feat}.png")
            try:
                plot_boxplot(df, cluster_col, feat, outfile, shortname)
                created.append(outfile)
            except Exception as e:
                print(f"Skipping {feat} for {shortname}: {e}")

    print(f"Done. Created {len(created)} images under {base_out}.")


if __name__ == '__main__':
    main()
