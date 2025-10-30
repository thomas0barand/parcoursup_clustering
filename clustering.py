import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Select categorical features to include
CATEGORICAL_FEATURES = [
    "Département de l’établissement"
    # "Filière de formation",
    # "Filière de formation très agrégée",
    # "Filière de formation détaillée bis"
]

NUMERICAL_FEATURES = [
'f_ratio_candidats',
'f_ratio_admis',
'f_selectivity_candidats',
'b_ratio_candidats',
'b_ratio_admis',
#'b_selectivity_candidats',
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
'taux_acces_ratio']


def load_and_prepare_data(data_path, numerical_features, categorical_features):
    """Load data and prepare features for clustering."""
    
    df = pd.read_csv(data_path)
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Remove columns beginning with "gen" or "tech"
    filtered_numerical_cols = [col for col in numerical_cols 
                               if not (col.startswith("gen") or col.startswith("tech"))]
    print(f"\nNumerical features: {len(filtered_numerical_cols)}")
    
    
    
    # One-hot encode categorical features
    df_categorical_encoded = pd.get_dummies(df[categorical_features], 
                                           prefix=categorical_features, 
                                           drop_first=True)
    print(f"Categorical features (one-hot encoded): {len(df_categorical_encoded.columns)}")
    print(f"Total features for clustering: {len(filtered_numerical_cols) + len(df_categorical_encoded.columns)}")
    
    # Combine numerical and encoded categorical features
    X_numerical = df[numerical_features]
    X = X_numerical
    X = pd.concat([X_numerical, df_categorical_encoded], axis=1) #, df_categorical_encoded] t
    
    return df, X


def perform_clustering(X, n_clusters=2, random_state=42, method='kmeans', eps=0.5, min_samples=5):
    """Perform clustering."""
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        return kmeans, labels
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        return dbscan, labels
    else:
        raise ValueError(f"Invalid clustering method: {method}")


def print_metrics(X, labels, kmeans, method='kmeans'):
    """Print clustering metrics and statistics."""
    print("=" * 50)
    print("CLUSTERING METRICS")
    print("=" * 50)
    if method == 'kmeans':
        print(f"Number of clusters: {kmeans.n_clusters}")
        print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    elif method == 'dbscan':
        print(f"Number of clusters: {len(np.unique(labels))}")
        # print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
    else:
        raise ValueError(f"Invalid clustering method: {method}")


def perform_pca(X, n_components=2, random_state=42):
    """Perform PCA transformation."""
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return pca, X_pca


def create_interactive_plot(df, X_pca, pca, output_path='data/clustering_interactive.html'):
    """Create interactive Plotly scatter plot."""
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': df['cluster'].astype(str),
        'Établissement': df['Établissement'],
        'Filière': df['Filière de formation'],
        'Département': df["Département de l’établissement"],
    })
    
    fig_interactive = px.scatter(
        df_plot, 
        x='PC1', 
        y='PC2', 
        color='Cluster',
        hover_data=['Établissement', 'Filière', 'Département'],
        title='Interactive Clusters in 2D PCA Space',
        labels={
            'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'
        },
        color_discrete_sequence=['#440154', '#fde724']
    )
    fig_interactive.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig_interactive.write_html(output_path)
    
    print("\n" + "=" * 50)
    print(f"Interactive plot saved to '{output_path}'")
    print("Open it in your browser to explore!")
    print("=" * 50)


def create_static_visualizations(
    df,
    X,
    X_pca,
    pca,
    kmeans,
    labels,
    method='kmeans',
    output_path='data/clustering_visualization.png',
    output_dir='data/clustering_vis',
    show_quality_metrics=True,
):
    """Create static matplotlib visualizations and save each subplot as a separate file.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'cluster' column with cluster labels.
    X : pandas.DataFrame
        Feature matrix.
    X_pca : np.ndarray
        PCA-transformed coordinates of X (2D).
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    kmeans : sklearn.cluster.KMeans | None
        Trained KMeans instance when method == 'kmeans'. Can be None for other methods.
    labels : array-like
        Cluster labels aligned with rows of X.
    method : str
        One of {'kmeans', 'dbscan', 'hierarchical'}.
    output_path : str
        Path to save the combined figure.
    output_dir : str
        Directory to save individual figures.
    show_quality_metrics : bool
        If False, do not compute or display silhouette/CH/DB metrics.
    """
    import os

    # Ensure the output directory exists for individual plots
    os.makedirs(output_dir, exist_ok=True)

    # Helper: compute clustering quality metrics
    def _metrics_text(X_in, labels_in):
        if not show_quality_metrics:
            return ""
        try:
            unique = np.unique(labels_in)
            if len(unique) < 2:
                return "Silhouette: N/A\nCH: N/A\nDB: N/A"
            sil = silhouette_score(X_in, labels_in)
            ch = calinski_harabasz_score(X_in, labels_in)
            db = davies_bouldin_score(X_in, labels_in)
            return f"Silhouette: {sil:.3f}\nCH: {ch:.1f}\nDB: {db:.3f}"
        except Exception:
            return "Silhouette: N/A\nCH: N/A\nDB: N/A"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. PCA scatter plot with clusters
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'],
                          cmap='viridis', alpha=0.6, edgecolors='k', s=50)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Clusters in 2D PCA Space')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    # Metrics box on combined PCA (optional)
    metrics_str = _metrics_text(X, labels)
    if metrics_str:
        ax1.text(0.02, 0.98, metrics_str, transform=ax1.transAxes,
                 va='top', ha='left', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Save plot 1
    fig1, ax1_indiv = plt.subplots(figsize=(8, 6))
    scatter_indiv = ax1_indiv.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'],
                                      cmap='viridis', alpha=0.6, edgecolors='k', s=50)
    ax1_indiv.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1_indiv.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1_indiv.set_title('Clusters in 2D PCA Space')
    plt.colorbar(scatter_indiv, ax=ax1_indiv, label='Cluster')
    # Metrics box on individual PCA (optional)
    if metrics_str:
        ax1_indiv.text(0.02, 0.98, metrics_str, transform=ax1_indiv.transAxes,
                       va='top', ha='left', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, '01_pca_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. Cluster distribution bar plot
    ax2 = axes[0, 1]
    cluster_counts = df['cluster'].value_counts().sort_index()
    ax2.bar(cluster_counts.index, cluster_counts.values, color='steelblue', edgecolor='k')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of samples')
    ax2.set_title('Cluster Distribution')
    ax2.set_xticks(cluster_counts.index)

    # Save plot 2
    fig2, ax2_indiv = plt.subplots(figsize=(8, 6))
    ax2_indiv.bar(cluster_counts.index, cluster_counts.values, color='steelblue', edgecolor='k')
    ax2_indiv.set_xlabel('Cluster')
    ax2_indiv.set_ylabel('Number of samples')
    ax2_indiv.set_title('Cluster Distribution')
    ax2_indiv.set_xticks(cluster_counts.index)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, '02_cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 3. Feature importance - Difference between cluster centers
    ax3 = axes[1, 0]
    if method == 'kmeans' and kmeans is not None:
        cluster_centers = kmeans.cluster_centers_
        if cluster_centers.shape[0] < 2:
            feature_importance = np.zeros(X.shape[1])
        else:
            feature_importance = np.abs(cluster_centers[0] - cluster_centers[1])
        all_feature_names = X.columns
        top_n = 10
        top_features_idx = np.argsort(feature_importance)[-top_n:]
        top_features = all_feature_names[top_features_idx]
        top_importance = feature_importance[top_features_idx]

        ax3.barh(range(top_n), top_importance, color='coral', edgecolor='k')
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels(top_features, fontsize=7)
        ax3.set_xlabel('Absolute Difference Between Cluster Centers')
        ax3.set_title(f'Top {top_n} Most Important Features for Clustering')
        ax3.grid(axis='x', alpha=0.3)
    elif method == 'dbscan':
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        if len(unique_labels) > 1:
            feature_importance = np.zeros(X.shape[1])
            for feature_idx in range(X.shape[1]):
                feature_values = X.iloc[:, feature_idx].values
                cluster_means = []
                for cluster_id in unique_labels:
                    cluster_mask = labels == cluster_id
                    if np.sum(cluster_mask) > 0:
                        cluster_means.append(np.mean(feature_values[cluster_mask]))
                if len(cluster_means) > 1:
                    feature_importance[feature_idx] = np.var(cluster_means)
                else:
                    feature_importance[feature_idx] = 0

            all_feature_names = X.columns
            top_n = 10
            top_features_idx = np.argsort(feature_importance)[-top_n:]
            top_features = all_feature_names[top_features_idx]
            top_importance = feature_importance[top_features_idx]

            ax3.barh(range(top_n), top_importance, color='coral', edgecolor='k')
            ax3.set_yticks(range(top_n))
            ax3.set_yticklabels(top_features, fontsize=7)
            ax3.set_xlabel('Variance Between Cluster Means')
            ax3.set_title(f'Top {top_n} Most Important Features for DBSCAN Clustering')
            ax3.grid(axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient clusters for\nfeature importance analysis',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Feature Importance (DBSCAN)')
    else:
        # For hierarchical or unsupported methods, show a placeholder
        ax3.text(0.5, 0.5, 'Feature importance not applicable',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Feature Importance')

    # Save plot 3
    fig3, ax3_indiv = plt.subplots(figsize=(8, 6))
    if method == 'kmeans' and kmeans is not None:
        if cluster_centers.shape[0] < 2:
            ax3_indiv.text(0.5, 0.5, 'Not enough clusters', ha='center', va='center', fontsize=12)
        else:
            ax3_indiv.barh(range(top_n), top_importance, color='coral', edgecolor='k')
            ax3_indiv.set_yticks(range(top_n))
            ax3_indiv.set_yticklabels(top_features, fontsize=7)
            ax3_indiv.set_xlabel('Absolute Difference Between Cluster Centers')
            ax3_indiv.set_title(f'Top {top_n} Most Important Features for Clustering')
            ax3_indiv.grid(axis='x', alpha=0.3)
    elif method == 'dbscan':
        if (len(unique_labels) <= 1):
            ax3_indiv.text(0.5, 0.5, 'Insufficient clusters for\nfeature importance analysis',
                           ha='center', va='center', fontsize=12)
            ax3_indiv.set_title('Feature Importance (DBSCAN)')
        else:
            ax3_indiv.barh(range(top_n), top_importance, color='coral', edgecolor='k')
            ax3_indiv.set_yticks(range(top_n))
            ax3_indiv.set_yticklabels(top_features, fontsize=7)
            ax3_indiv.set_xlabel('Variance Between Cluster Means')
            ax3_indiv.set_title(f'Top {top_n} Most Important Features for DBSCAN Clustering')
            ax3_indiv.grid(axis='x', alpha=0.3)
    else:
        ax3_indiv.text(0.5, 0.5, 'Feature importance not applicable',
                       ha='center', va='center', fontsize=12)
        ax3_indiv.set_title('Feature Importance')
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, '03_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 4. Heatmap of cluster centers for top features
    ax4 = axes[1, 1]
    if method == 'kmeans' and kmeans is not None:
        if cluster_centers.shape[0] < 2:
            ax4.text(0.5, 0.5, 'Not enough clusters', ha='center', va='center', fontsize=12)
            ax4.set_title('Cluster Centers Heatmap (KMeans)')
        else:
            cluster_centers_top = cluster_centers[:, top_features_idx]
            sns.heatmap(cluster_centers_top, annot=True, fmt='.2f', cmap='RdYlGn',
                        xticklabels=top_features, yticklabels=[f'Cluster {i}' for i in range(kmeans.n_clusters)],
                        ax=ax4, cbar_kws={'label': 'Value'})
            ax4.set_title(f'Cluster Centers Heatmap (Top {top_n} Features)')
            ax4.set_xlabel('Features')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    elif method == 'dbscan':
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        if len(unique_labels) > 1 and 'top_features_idx' in locals():
            cluster_means_matrix = np.zeros((len(unique_labels), len(top_features_idx)))
            for i, cluster_id in enumerate(unique_labels):
                cluster_mask = labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_data = X.iloc[cluster_mask, top_features_idx]
                    cluster_means_matrix[i] = cluster_data.mean().values

            sns.heatmap(cluster_means_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                        xticklabels=top_features, yticklabels=[f'Cluster {int(cluster_id)}' for cluster_id in unique_labels],
                        ax=ax4, cbar_kws={'label': 'Mean Value'})
            ax4.set_title(f'Cluster Means Heatmap (Top {top_n} Features)')
            ax4.set_xlabel('Features')
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        else:
            ax4.text(0.5, 0.5, 'Insufficient clusters for\nheatmap visualization',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Cluster Centers Heatmap (DBSCAN)')
    else:
        ax4.text(0.5, 0.5, 'Heatmap not applicable',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Cluster Centers Heatmap')

    # Save plot 4
    fig4, ax4_indiv = plt.subplots(figsize=(8, 6))
    if method == 'kmeans' and kmeans is not None:
        if cluster_centers.shape[0] < 2:
            ax4_indiv.text(0.5, 0.5, 'Not enough clusters', ha='center', va='center', fontsize=12)
            ax4_indiv.set_title('Cluster Centers Heatmap (KMeans)')
        else:
            cluster_centers_top = cluster_centers[:, top_features_idx]
            sns.heatmap(cluster_centers_top, annot=True, fmt='.2f', cmap='RdYlGn',
                        xticklabels=top_features, yticklabels=[f'Cluster {i}' for i in range(kmeans.n_clusters)],
                        ax=ax4_indiv, cbar_kws={'label': 'Value'})
            ax4_indiv.set_title(f'Cluster Centers Heatmap (Top {top_n} Features)')
            ax4_indiv.set_xlabel('Features')
            plt.setp(ax4_indiv.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    elif method == 'dbscan':
        if len(unique_labels) > 1 and 'top_features_idx' in locals():
            cluster_means_matrix = np.zeros((len(unique_labels), len(top_features_idx)))
            for i, cluster_id in enumerate(unique_labels):
                cluster_mask = labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_data = X.iloc[cluster_mask, top_features_idx]
                    cluster_means_matrix[i] = cluster_data.mean().values

            sns.heatmap(cluster_means_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                        xticklabels=top_features, yticklabels=[f'Cluster {int(cluster_id)}' for cluster_id in unique_labels],
                        ax=ax4_indiv, cbar_kws={'label': 'Mean Value'})
            ax4_indiv.set_title(f'Cluster Means Heatmap (Top {top_n} Features)')
            ax4_indiv.set_xlabel('Features')
            plt.setp(ax4_indiv.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        else:
            ax4_indiv.text(0.5, 0.5, 'Insufficient clusters for\nheatmap visualization',
                           ha='center', va='center', fontsize=12)
            ax4_indiv.set_title('Cluster Centers Heatmap (DBSCAN)')
    else:
        ax4_indiv.text(0.5, 0.5, 'Heatmap not applicable',
                       ha='center', va='center', fontsize=12)
        ax4_indiv.set_title('Cluster Centers Heatmap')
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, '04_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("\n" + "=" * 50)
    print(f"Visualization saved to '{output_path}'")
    print(f"Individual plots saved to '{output_dir}/'")
    print("=" * 50)
    plt.show()
    # Do not return silhouette to avoid forcing its computation
    return None


def main(n_clusters=2):
    """Main function to orchestrate the clustering pipeline."""
    # Load and prepare data
    df, X = load_and_prepare_data(data_path='data/df_features_normalized.csv', numerical_features=NUMERICAL_FEATURES, categorical_features=CATEGORICAL_FEATURES)
    
    # Perform clustering
    kmeans, labels = perform_clustering(X, n_clusters=n_clusters)
    df['cluster'] = labels
    
    # Save results
    df.to_csv('data/df_clustering.csv', index=False)
    
    # Print metrics
    print_metrics(X, labels, kmeans)
    
    # Perform PCA
    pca, X_pca = perform_pca(X)
    
    # Create visualizations
    create_interactive_plot(df, X_pca, pca)
    create_static_visualizations(df, X, X_pca, pca, kmeans, labels)


if __name__ == "__main__":
    main(n_clusters=5)
