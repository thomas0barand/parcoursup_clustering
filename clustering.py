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


def perform_clustering(X, n_clusters=2, random_state=42):
    """Perform K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def print_metrics(X, labels, kmeans):
    """Print clustering metrics and statistics."""
    print("=" * 50)
    print("CLUSTERING METRICS")
    print("=" * 50)
    print(f"Number of clusters: {kmeans.n_clusters}")
    print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
    
    print("\n" + "=" * 50)
    print("CLUSTER DISTRIBUTION")
    print("=" * 50)
    print(pd.Series(labels).value_counts().sort_index())
    
    print("\n" + "=" * 50)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 50)
    all_feature_names = X.columns
    feature_importance_all = np.abs(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])
    top_10_idx = np.argsort(feature_importance_all)[-10:][::-1]
    for i, idx in enumerate(top_10_idx, 1):
        print(f"{i}. {all_feature_names[idx]}: {feature_importance_all[idx]:.4f}")


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


def create_static_visualizations(df, X, X_pca, pca, kmeans, output_path='data/clustering_visualization.png'):
    """Create static matplotlib visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. PCA scatter plot with clusters
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], 
                          cmap='viridis', alpha=0.6, edgecolors='k', s=50)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('Clusters in 2D PCA Space')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 2. Cluster distribution bar plot
    ax2 = axes[0, 1]
    cluster_counts = df['cluster'].value_counts().sort_index()
    ax2.bar(cluster_counts.index, cluster_counts.values, color='steelblue', edgecolor='k')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Number of samples')
    ax2.set_title('Cluster Distribution')
    ax2.set_xticks(cluster_counts.index)
    
    # 3. Feature importance - Difference between cluster centers
    ax3 = axes[1, 0]
    cluster_centers = kmeans.cluster_centers_
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
    
    # 4. Heatmap of cluster centers for top features
    ax4 = axes[1, 1]
    cluster_centers_top = cluster_centers[:, top_features_idx]
    sns.heatmap(cluster_centers_top, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=top_features, yticklabels=[f'Cluster {i}' for i in range(kmeans.n_clusters)],
                ax=ax4, cbar_kws={'label': 'Value'})
    ax4.set_title(f'Cluster Centers Heatmap (Top {top_n} Features)')
    ax4.set_xlabel('Features')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("\n" + "=" * 50)
    print(f"Visualization saved to '{output_path}'")
    print("=" * 50)
    plt.show()


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
    create_static_visualizations(df, X, X_pca, pca, kmeans)


if __name__ == "__main__":
    main(n_clusters=5)
