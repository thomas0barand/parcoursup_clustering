"""Utilities for evaluating distance metrics prior to hierarchical clustering.

This module focuses on the first step of the hierarchical clustering pipeline:
verifying that the dissimilarity (distance) measures produce cophenetic distances
that correlate well with the original pairwise distances. The distance metric
with the highest cophenetic correlation coefficient is usually a good candidate
to feed into the actual hierarchical clustering step.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.cluster.hierarchy import (
    cophenet,
    dendrogram,
    fcluster,
    inconsistent,
    linkage,
)
from scipy.spatial.distance import pdist

from main import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES
)

from clustering import load_and_prepare_data, perform_pca
# Distance metrics to evaluate. This reproduces the list provided by the user.
DISTANCE_METRICS: Tuple[str, ...] = (
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "yule",
)

# Metrics that require binary (0/1) inputs.
BINARY_METRICS: Tuple[str, ...] = (
    "dice",
    "jaccard",
    "kulsinski",
    "matching",
    "rogerstanimoto",
    "russellrao",
    "sokalmichener",
    "sokalsneath",
    "yule",
)

# Metrics that require rows to be non-negative and sum to 1.
PROBABILITY_METRICS: Tuple[str, ...] = ("jensenshannon",)


DEFAULT_CLUSTER_COLORS: Tuple[str, ...] = (
    "#9B5DE5",
    "#d62828",
    "#a7c957",
    "#00F5D4",
    "#f8961e",
    "#577590",
    "#ffb703",
    "#8ecae6",
)


def _load_feature_matrix(
    data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load data, features, and a numpy matrix ready for distance computations."""

    df, features = load_and_prepare_data(
        data_path=data_path,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    matrix = features.to_numpy(dtype=np.float64, copy=True)

    # Ensure we do not carry over rows with NaNs or infs into pdist.
    finite_row_mask = np.isfinite(matrix).all(axis=1)
    if not finite_row_mask.all():
        df = df.loc[finite_row_mask].reset_index(drop=True)
        features = features.loc[finite_row_mask].reset_index(drop=True)
        matrix = matrix[finite_row_mask]

    return df, features, matrix


def _is_binary_matrix(matrix: np.ndarray) -> bool:
    """Return True if the matrix only contains binary values (0/1)."""

    unique_values = np.unique(matrix)
    return np.array_equal(unique_values, np.array([0.0, 1.0]))


def _is_probability_matrix(matrix: np.ndarray, atol: float = 1e-6) -> bool:
    """Return True if the matrix contains probability distributions per row."""

    if np.any(matrix < 0):
        return False
    row_sums = matrix.sum(axis=1)
    return np.allclose(row_sums, 1.0, atol=atol)


def _metric_kwargs(metric: str, matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Return additional keyword arguments required by specific metrics."""

    if metric == "seuclidean":
        variance = np.var(matrix, axis=0, ddof=1)
        variance = np.where(variance == 0, 1e-12, variance)
        return {"V": variance}
    if metric == "mahalanobis":
        covariance = np.cov(matrix, rowvar=False)
        inverse_covariance = np.linalg.pinv(covariance)
        return {"VI": inverse_covariance}
    return {}


def evaluate_distance_metrics(
    matrix: np.ndarray,
    metrics: Iterable[str],
    linkage_method: str = "average",
    allow_incompatible: bool = False,
) -> Tuple[List[Tuple[str, float]], Dict[str, str]]:
    """Compute the cophenetic correlation coefficient for each metric.

    Parameters
    ----------
    matrix : np.ndarray
        Numeric feature matrix where rows represent samples.
    metrics : Iterable[str]
        Iterable of metric names accepted by :func:`scipy.spatial.distance.pdist`.
    linkage_method : str, optional
        Linkage strategy to use when building the dendrogram. Defaults to
        ``"average"`` which supports any dissimilarity metric.
    allow_incompatible : bool, optional
        If False (default), metrics that are incompatible with the data type
        (e.g. binary-only metrics) are skipped with an explanation instead of
        forcing a computation that would fail or produce meaningless values.

    Returns
    -------
    results : list[tuple[str, float]]
        List of (metric, cophenetic correlation coefficient) tuples sorted in
        descending order of the coefficient.
    skipped : dict[str, str]
        Mapping of metric name to the reason it was skipped or failed.
    """

    results: List[Tuple[str, float]] = []
    skipped: Dict[str, str] = {}

    is_binary_input = _is_binary_matrix(matrix)
    is_probability_input = _is_probability_matrix(matrix)

    for metric in metrics:
        # Skip metrics that are known to be incompatible with this dataset.
        if not allow_incompatible and metric in BINARY_METRICS and not is_binary_input:
            skipped[metric] = "requires binary (0/1) input"
            continue
        if not allow_incompatible and metric in PROBABILITY_METRICS and not is_probability_input:
            skipped[metric] = "requires non-negative rows that sum to 1"
            continue
        if linkage_method == "ward" and metric != "euclidean":
            skipped[metric] = "ward linkage is only defined with Euclidean distances"
            continue

        try:
            metric_kwargs = _metric_kwargs(metric, matrix)
            condensed_distances = pdist(matrix, metric=metric, **metric_kwargs)

            if not np.isfinite(condensed_distances).all():
                skipped[metric] = "produced non-finite distances"
                continue
            if np.allclose(condensed_distances, 0.0):
                skipped[metric] = "produced zero-valued distances"
                continue

            linkage_matrix = linkage(condensed_distances, method=linkage_method)
            cophenetic_corr, _ = cophenet(linkage_matrix, condensed_distances)
            results.append((metric, float(cophenetic_corr)))
        except Exception as exc:  # pragma: no cover - defensive programming
            skipped[metric] = f"error during evaluation ({exc})"

    results.sort(key=lambda item: item[1], reverse=True)
    return results, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate distance metrics for hierarchical clustering, "
            "build the dendrogram, and optionally generate cluster visualisations."
        )
    )
    parser.add_argument(
        "--data-path",
        default="data/df_features_normalized.csv",
        help="Path to the normalized feature dataset produced by main.py.",
    )
    parser.add_argument(
        "--clean-data-path",
        default="data/df_features_cleaned.csv",
        help="Path to the cleaned (non-normalised) feature dataset used for descriptive statistics.",
    )
    parser.add_argument(
        "--linkage",
        default="average",
        choices=(
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        ),
        help="Hierarchical linkage criterion to use.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DISTANCE_METRICS),
        help="Explicit list of distance metrics to evaluate.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Force the metric used for the final linkage/dendrogram generation. "
            "Defaults to the best metric according to cophenetic correlation."
        ),
    )
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help=(
            "Attempt to run metrics even if they are likely incompatible with the "
            "data (may raise errors)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="If > 0, only display the top-k metrics by cophenetic correlation.",
    )
    parser.add_argument(
        "--linkage-path",
        default="data/hierarchical_linkage.npy",
        help="Where to save the linkage matrix produced for the selected metric.",
    )
    parser.add_argument(
        "--dendrogram-path",
        default="data/hierarchical_dendrogram.png",
        help="Where to save the primary dendrogram plot.",
    )
    parser.add_argument(
        "--dendrogram-orientation",
        default="top",
        choices=("top", "bottom", "left", "right"),
        help="Orientation for the primary dendrogram visualisation.",
    )
    parser.add_argument(
        "--dendrogram-color-threshold",
        type=float,
        default=None,
        help="Optional colour threshold to pass to scipy.dendrogram.",
    )
    parser.add_argument(
        "--dendrogram-distance-sort",
        choices=("ascending", "descending"),
        default=None,
        help="Optional distance sorting strategy for the dendrogram (ascending/descending).",
    )
    parser.add_argument(
        "--dendrogram-show-leaf-counts",
        action="store_true",
        help="Include leaf counts next to each cluster in the dendrogram.",
    )
    parser.add_argument(
        "--truncate-dendrogram",
        type=int,
        default=30,
        help=(
            "If > 0, display only the last p merged clusters in the dendrogram "
            "(truncate_mode='lastp'). Set to 0 to show the full tree."
        ),
    )
    parser.add_argument(
        "--cut-distance",
        type=float,
        default=None,
        help="Distance threshold used to cut the dendrogram (criterion='distance').",
    )
    parser.add_argument(
        "--cut-maxclust",
        type=int,
        default=None,
        help="Maximum number of clusters for fcluster (criterion='maxclust').",
    )
    parser.add_argument(
        "--cluster-output",
        default="data/df_hierarchical_clusters.csv",
        help="Path where the dataframe with hierarchical cluster assignments is saved.",
    )
    parser.add_argument(
        "--cluster-dendrogram-path",
        default="data/hierarchical_dendrogram_clusters.png",
        help="Where to save the cluster-annotated dendrogram (orientation='right').",
    )
    parser.add_argument(
        "--interactive-plot-path",
        default="data/hierarchical_interactive.html",
        help="Where to save the interactive Plotly visualisation.",
    )
    parser.add_argument(
        "--static-plots-dir",
        default="data/hierarchical_group_plots",
        help="Directory where static descriptive plots will be written.",
    )
    parser.add_argument(
        "--skip-interactive",
        action="store_true",
        help="Skip generation of the Plotly interactive scatter plot.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip creation of descriptive statistics plots.",
    )
    parser.add_argument(
        "--inconsistency-depth",
        type=int,
        default=2,
        help="Depth parameter passed to scipy.cluster.hierarchy.inconsistent.",
    )
    parser.add_argument(
        "--inconsistency-top-k",
        type=int,
        default=8,
        help=(
            "Number of merges with highest inconsistency coefficients to display "
            "(suggests potential cluster counts)."
        ),
    )
    parser.add_argument(
        "--inconsistency-tail",
        type=int,
        default=10,
        help=(
            "Number of final merges (closest to the root) to display in the "
            "inconsistency tail report."
        ),
    )
    parser.add_argument(
        "--inconsistency-window",
        type=int,
        default=3,
        help=(
            "Number of neighbouring merges below a candidate merge to use when "
            "assessing inconsistency ratios in the tail."
        ),
    )
    parser.add_argument(
        "--inconsistency-threshold",
        type=float,
        default=1.05,
        help=(
            "Minimum ratio between a tail merge inconsistency coefficient and "
            "the average of its neighbours required to flag a potential cut."
        ),
    )
    return parser.parse_args()


def _build_linkage(
    matrix: np.ndarray,
    metric: str,
    linkage_method: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute condensed distances and the linkage matrix for the dataset."""

    metric_kwargs = _metric_kwargs(metric, matrix)
    condensed_distances = pdist(matrix, metric=metric, **metric_kwargs)
    linkage_matrix = linkage(condensed_distances, method=linkage_method)
    return condensed_distances, linkage_matrix


def _save_linkage_matrix(linkage_matrix: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, linkage_matrix)


def _plot_and_save_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Path,
    truncate_p: int,
    title: str,
    orientation: str = "top",
    labels: Sequence[str] | None = None,
    color_threshold: float | None = None,
    cut_distance: float | None = None,
    show_leaf_counts: bool = False,
    distance_sort: str | None = None,
    figsize: Tuple[int, int] = (14, 7),
    leaf_color_map: Dict[str, str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if color_threshold is None and cut_distance is not None:
        color_threshold = cut_distance

    plt.figure(figsize=figsize)
    dendrogram_kwargs: Dict[str, object] = {
        "orientation": orientation,
        "color_threshold": color_threshold,
        "show_leaf_counts": show_leaf_counts,
    }
    if labels is not None:
        dendrogram_kwargs["labels"] = labels
        dendrogram_kwargs["no_labels"] = False
    else:
        dendrogram_kwargs["no_labels"] = True

    if truncate_p and truncate_p > 0:
        dendrogram_kwargs.update({"truncate_mode": "lastp", "p": truncate_p})

    if distance_sort is not None:
        dendrogram_kwargs["distance_sort"] = distance_sort

    dendro = dendrogram(linkage_matrix, **dendrogram_kwargs)
    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Merged items")

    if cut_distance is not None:
        plt.axvline(x=cut_distance, ymin=0.0, ymax=1.0, color="red", linewidth=3)

    # If a leaf color mapping is provided, color tick labels to match clusters
    if leaf_color_map:
        ax = plt.gca()
        # Depending on orientation, leaf labels are on x or y axis
        if orientation in ("top", "bottom"):
            ticklabels = ax.get_xticklabels()
        else:
            ticklabels = ax.get_yticklabels()
        for lbl in ticklabels:
            text = lbl.get_text()
            if text in leaf_color_map:
                lbl.set_color(leaf_color_map[text])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _assign_clusters(
    linkage_matrix: np.ndarray,
    cut_distance: float | None = None,
    cut_maxclust: int | None = None,
) -> np.ndarray | None:
    """Assign cluster labels using scipy's fcluster helper."""

    if cut_distance is not None and cut_maxclust is not None:
        raise ValueError("Specify either --cut-distance or --cut-maxclust, not both.")

    if cut_distance is not None:
        if cut_distance <= 0:
            raise ValueError("--cut-distance must be strictly positive.")
        return fcluster(linkage_matrix, t=cut_distance, criterion="distance")

    if cut_maxclust is not None:
        if cut_maxclust <= 1:
            raise ValueError("--cut-maxclust must be greater than 1.")
        return fcluster(linkage_matrix, t=cut_maxclust, criterion="maxclust")

    return None


def _build_color_palette(n_clusters: int) -> List[str]:
    if n_clusters <= 0:
        return []

    base = list(DEFAULT_CLUSTER_COLORS)
    if n_clusters <= len(base):
        return base[:n_clusters]

    dynamic_palette = sns.color_palette("husl", n_clusters)
    return [to_hex(color) for color in dynamic_palette]


def _build_cluster_color_map(cluster_labels: np.ndarray) -> Dict[str, str]:
    """Create a stable mapping from cluster id (as string) to color.

    Clusters are sorted ascending before mapping onto the palette to ensure
    consistent references across plots (PCA, dendrogram, group stats).
    """
    unique_clusters_sorted = np.sort(np.unique(cluster_labels))
    palette = _build_color_palette(len(unique_clusters_sorted))
    return {str(cluster): palette[idx] for idx, cluster in enumerate(unique_clusters_sorted)}


def _analyze_inconsistency(
    matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    depth: int,
    top_k: int,
    tail_count: int,
    window: int,
    threshold: float,
) -> None:
    depth = max(depth, 1)
    window = max(window, 1)
    threshold = max(threshold, 0.0)
    stats = inconsistent(linkage_matrix, d=depth)
    coefficients = stats[:, 3]
    total_merges = coefficients.shape[0]
    if total_merges == 0:
        print("No inconsistency statistics available (insufficient samples).")
        return

    n_samples = matrix.shape[0]
    merges = np.arange(total_merges)
    clusters_after_merge = np.maximum(1, n_samples - (merges + 1))

    print(
        "\nInconsistency analysis"
        f" (depth={depth}, merges={total_merges})"
    )

    global_counts: Set[int] = set()
    tail_counts: Set[int] = set()

    if top_k > 0:
        top_indices = np.argsort(coefficients)[-top_k:][::-1]

        print(f"Top {len(top_indices)} inconsistency coefficients:")
        header = f"{'rank':>4} | {'coef':>10} | {'clusters_after':>15} | {'merge_idx':>10}"
        print(header)
        print("-" * len(header))
        for rank, idx in enumerate(top_indices, start=1):
            print(
                f"{rank:>4} | {coefficients[idx]:10.6f} | "
                f"{clusters_after_merge[idx]:15d} | {idx:10d}"
            )
            if clusters_after_merge[idx] >= 2:
                global_counts.add(int(clusters_after_merge[idx]))

    tail_entries: List[Tuple[int, float, int]] = []
    if tail_count > 0:
        start_idx = max(0, total_merges - tail_count)
        tail_indices = list(range(start_idx, total_merges))
        if tail_indices:
            print(
                f"\nTail inconsistency coefficients "
                f"(last {len(tail_indices)} merges, root first):"
            )
            print(f"{'clusters_after':>15} | {'coef':>10} | {'merge_idx':>10}")
            print("-" * 47)
            for idx in reversed(tail_indices):
                cluster_count = int(clusters_after_merge[idx])
                coefficient = float(coefficients[idx])
                print(
                    f"{cluster_count:15d} | "
                    f"{coefficient:10.6f} | {idx:10d}"
                )
                tail_entries.append((cluster_count, coefficient, idx))

    if tail_entries:
        highlight_rows: List[Tuple[int, float, float]] = []
        effective_window = min(window, max(len(tail_entries) - 1, 1))
        if effective_window >= 1:
            for i, (cluster_count, coeff_value, _) in enumerate(tail_entries):
                neighbours = tail_entries[i + 1 : i + 1 + effective_window]
                if not neighbours:
                    continue
                baseline = float(np.mean([entry[1] for entry in neighbours]))
                if baseline <= 0:
                    continue
                ratio = coeff_value / baseline
                if ratio >= threshold:
                    highlight_rows.append((cluster_count, coeff_value, ratio))
                    if cluster_count >= 2:
                        tail_counts.add(cluster_count)

        if highlight_rows:
            print(
                f"\nTail highlights (ratio ≥ {threshold:.2f}, window={effective_window}):"
            )
            header = f"{'clusters_after':>15} | {'coef':>10} | {'ratio':>8}"
            print(header)
            print("-" * len(header))
            for cluster_count, coeff_value, ratio in highlight_rows:
                print(
                    f"{cluster_count:15d} | {coeff_value:10.6f} | {ratio:8.3f}"
                )

    if global_counts:
        ordered_counts = sorted(global_counts)
        suggestions = ", ".join(str(n) for n in ordered_counts)
        print(
            "\nSuggested cluster counts from global inconsistency peaks: "
            f"{suggestions}"
        )

    if tail_counts:
        ordered_tail = sorted(tail_counts)
        tail_suggestions = ", ".join(str(n) for n in ordered_tail)
        print(
            "Suggested cluster counts from tail analysis: "
            f"{tail_suggestions}"
        )

    if global_counts or tail_counts:
        combined = sorted(global_counts.union(tail_counts))
        combined_suggestions = ", ".join(str(n) for n in combined)
        print(
            "Combined suggested cluster counts: "
            f"{combined_suggestions}"
        )


def _create_interactive_plot(
    df_with_meta: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    cluster_labels: np.ndarray,
    output_path: Path,
) -> None:
    # Stable color assignment shared across visuals
    color_map = _build_cluster_color_map(cluster_labels)
    unique_clusters_sorted = np.sort(np.unique(cluster_labels))

    pca, X_pca = perform_pca(feature_matrix, n_components=2)
    df_plot = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Cluster": cluster_labels.astype(str),
            "Établissement": df_with_meta["Établissement"],
            "Filière": df_with_meta["Filière de formation"],
            "Département": df_with_meta["Département de l’établissement"],
        }
    )

    fig = px.scatter(
        df_plot,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Établissement", "Filière", "Département"],
        title=(
            "Hierarchical clusters projected in 2D PCA space"
        ),
        # Explicit mapping guarantees consistent colors with other visuals
        color_discrete_map=color_map,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig.update_layout(
        legend_title_text="Cluster",
        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)",
        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Interactive plot saved to '{output_path}'.")

    # Also save a static PNG using Matplotlib (no extra deps like kaleido)
    png_path = output_path.with_suffix(".png")
    plt.figure(figsize=(10, 7))
    point_colors = [color_map[str(lbl)] for lbl in cluster_labels.astype(str)]
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=point_colors,
        s=36,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.9,
    )
    # Legend per cluster
    handles = []
    labels = []
    for cluster in unique_clusters_sorted:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[str(cluster)],
                                  markeredgecolor="white", markeredgewidth=0.5, markersize=9, linestyle=''))
        labels.append(str(cluster))
    plt.legend(handles, labels, title="Cluster", loc="best")
    plt.title("Hierarchical clusters projected in 2D PCA space")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Static PCA plot saved to '{png_path}'.")


def _create_group_statistics(
    df_with_clusters: pd.DataFrame,
    palette: List[str],
    output_dir: Path,
) -> None:
    if "Group" not in df_with_clusters.columns:
        print("Group column missing from dataframe; skipping statistical plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_order = sorted(df_with_clusters["Group"].dropna().unique())
    if not cluster_order:
        print("No clusters detected; skipping statistical plots.")
        return

    palette_sequence = [palette[i % len(palette)] for i in range(len(cluster_order))]

    sns.set_theme(style="whitegrid")

    # 1. Boxplot of applicant volume
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_with_clusters,
        y="Group",
        x="Effectif total des candidats en phase principale",
        palette=palette_sequence,
        order=cluster_order,
        orient="h",
        ax=ax,
    )
    ax.set_title("Effectif total des candidats en phase principale par groupe")
    ax.set_xlabel("Effectif total des candidats en phase principale")
    ax.set_ylabel("Groupes")
    fig.tight_layout()
    path_wishes = output_dir / "group_wishes_boxplot.png"
    fig.savefig(path_wishes, dpi=300)
    plt.close(fig)

    # 2. Histograms for candidate types
    hist_cols = [
        "gen_admis_ratio",
        "tech_admis_ratio",
        "prof_admis_ratio",
    ]
    hist_labels = [
        "Admis issus d'un bac général (ratio)",
        "Admis issus d'un bac technologique (ratio)",
        "Admis issus d'un bac professionnel (ratio)",
    ]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    for col, label, ax in zip(hist_cols, hist_labels, axes):
        sns.histplot(
            data=df_with_clusters,
            x=col,
            hue="Group",
            hue_order=cluster_order,
            palette=palette_sequence,
            multiple="stack",
            ax=ax,
        )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("Effectifs")
    fig.suptitle("Répartition des admis par type de baccalauréat", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path_hist = output_dir / "group_bac_histograms.png"
    fig.savefig(path_hist, dpi=300)
    plt.close(fig)

    admissions_summary = df_with_clusters.groupby("Group")[hist_cols].mean()
    print("\nMoyennes des ratios d'admis par groupe:")
    print(admissions_summary)

    # 3. Capacity vs group boxplot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_with_clusters,
        y="Group",
        x="Capacité de l’établissement par formation",
        palette=palette_sequence,
        order=cluster_order,
        orient="h",
        ax=ax,
    )
    ax.set_title("Capacité par formation selon le groupe")
    ax.set_xlabel("Capacité de l’établissement par formation")
    ax.set_ylabel("Groupes")
    fig.tight_layout()
    path_capacity = output_dir / "group_capacity_boxplot.png"
    fig.savefig(path_capacity, dpi=300)
    plt.close(fig)

    # 4. Female admission ratio summary
    female_means = df_with_clusters.groupby("Group")["f_ratio_admis"].mean() * 100
    print("\nProportion moyenne de femmes admises par groupe (%):")
    print(female_means.round(2))

    # 5. Access rate boxplot and histogram
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    sns.boxplot(
        data=df_with_clusters,
        y="Group",
        x="taux_acces_ratio",
        palette=palette_sequence,
        order=cluster_order,
        orient="h",
        ax=axes[0],
    )
    axes[0].set_title("Taux d'accès ratio par groupe")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Groupes")

    sns.histplot(
        data=df_with_clusters,
        x="taux_acces_ratio",
        hue="Group",
        hue_order=cluster_order,
        palette=palette_sequence,
        multiple="stack",
        ax=axes[1],
    )
    axes[1].set_xlabel("Taux d'accès ratio")
    axes[1].set_ylabel("Effectifs")
    fig.suptitle("Répartition du taux d'accès par groupe", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path_access = output_dir / "group_access_rate.png"
    fig.savefig(path_access, dpi=300)
    plt.close(fig)

    # 6. Mentions scatterplots
    mention_pairs = [
        ("assez_bien_mention_ratio", "tres_bien_mention_ratio"),
        ("bien_mention_ratio", "tres_bien_mention_ratio"),
        ("assez_bien_mention_ratio", "bien_mention_ratio"),
    ]
    mention_labels = [
        ("Mention assez bien", "Mention très bien"),
        ("Mention bien", "Mention très bien"),
        ("Mention assez bien", "Mention bien"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for (x_col, y_col), (x_label, y_label), ax in zip(mention_pairs, mention_labels, axes):
        sns.scatterplot(
            data=df_with_clusters,
            x=x_col,
            y=y_col,
            hue="Group",
            hue_order=cluster_order,
            palette=palette_sequence,
            alpha=0.7,
            ax=ax,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    fig.suptitle("Répartition des mentions par groupe", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path_mentions = output_dir / "group_mentions_scatter.png"
    fig.savefig(path_mentions, dpi=300)
    plt.close(fig)

    # 7. Scholarship vs bachelor scatter per group
    n_clusters = len(cluster_order)
    ncols = min(3, n_clusters)
    nrows = math.ceil(n_clusters / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_array = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])
    for idx, cluster in enumerate(cluster_order):
        ax = axes_array[idx]
        subset = df_with_clusters[df_with_clusters["Group"] == cluster]
        sns.scatterplot(
            data=subset,
            x="meme_academie_ratio",
            y="meme_etablissement_ratio",
            color=palette_sequence[idx % len(palette_sequence)],
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(f"Groupe {cluster}")
        ax.set_xlabel("Ratio même académie")
        ax.set_ylabel("Ratio même établissement")

    for extra_ax in axes_array[len(cluster_order):]:
        extra_ax.remove()
    fig.suptitle("Origine géographique des admis par groupe", fontsize=16, fontweight="bold")
    fig.tight_layout()
    path_geo = output_dir / "group_geography_scatter.png"
    fig.savefig(path_geo, dpi=300)
    plt.close(fig)

def _load_clean_features_with_groups(
    df_with_clusters: pd.DataFrame,
    clean_data_path: str,
) -> pd.DataFrame | None:
    clean_path = Path(clean_data_path)
    if not clean_path.exists():
        print(f"Clean data file '{clean_path}' not found; skipping descriptive statistics.")
        return None

    clean_df = pd.read_csv(clean_path)

    join_candidates = [
        ["Code UAI de l'établissement", "Filière de formation détaillée bis"],
        ["Code UAI de l'établissement"],
    ]

    for join_cols in join_candidates:
        if not all(col in df_with_clusters.columns for col in join_cols):
            continue
        if not all(col in clean_df.columns for col in join_cols):
            continue

        mapping = df_with_clusters[join_cols + ["Group"]].drop_duplicates()
        merged = clean_df.merge(mapping, on=join_cols, how="left")
        matched = merged.dropna(subset=["Group"])
        if matched.empty:
            continue
        matched.sort_values(by=join_cols, inplace=True)
        return matched

    print("Unable to align clean features with cluster assignments; skipping descriptive statistics.")
    return None


def main() -> None:
    args = _parse_args()

    df, features, matrix = _load_feature_matrix(args.data_path)
    results, skipped = evaluate_distance_metrics(
        matrix=matrix,
        metrics=args.metrics,
        linkage_method=args.linkage,
        allow_incompatible=args.allow_incompatible,
    )

    top_k = args.top_k if args.top_k and args.top_k > 0 else len(results)

    print("\nCophenetic correlation coefficients")
    print(f"  linkage method : {args.linkage}")
    print(f"  feature count  : {features.shape[1]}\n")

    if results:
        header = f"{'metric':>20} | {'cophenetic_corr':>16}"
        print(header)
        print("-" * len(header))
        for metric, score in results[:top_k]:
            print(f"{metric:>20} | {score:16.6f}")
        best_metric, best_score = results[0]
        print(
            f"\nBest metric (by cophenetic correlation): "
            f"{best_metric} -> {best_score:.6f}"
        )
    else:
        print("No metrics produced valid cophenetic correlations.")

    if skipped:
        print("\nSkipped metrics:")
        for metric, reason in skipped.items():
            print(f"  {metric:>20} : {reason}")

    if not results:
        return

    results_dict = dict(results)
    selected_metric = args.metric or results[0][0]
    if selected_metric not in results_dict:
        raise ValueError(
            f"Requested metric '{selected_metric}' was not evaluated successfully."
        )

    if args.linkage == "ward" and selected_metric != "euclidean":
        raise ValueError(
            "Ward linkage requires the Euclidean metric. "
            "Change --linkage or --metric."
        )

    condensed_distances, linkage_matrix = _build_linkage(
        matrix=matrix,
        metric=selected_metric,
        linkage_method=args.linkage,
    )

    cophenetic_corr, _ = cophenet(linkage_matrix, condensed_distances)
    print(
        f"\nCophenetic correlation for linkage generated with '{selected_metric}': "
        f"{cophenetic_corr:.6f}"
    )

    linkage_path = Path(args.linkage_path)
    _save_linkage_matrix(linkage_matrix, linkage_path)
    print(f"Linkage matrix saved to '{linkage_path}'.")

    dendrogram_path = Path(args.dendrogram_path)
    truncate_p = max(args.truncate_dendrogram, 0)
    dendrogram_title = (
        f"Hierarchical clustering ({args.linkage} linkage, {selected_metric} metric)"
    )
    _plot_and_save_dendrogram(
        linkage_matrix=linkage_matrix,
        output_path=dendrogram_path,
        truncate_p=truncate_p,
        title=dendrogram_title,
        orientation=args.dendrogram_orientation,
        labels=None,
        color_threshold=args.dendrogram_color_threshold,
        cut_distance=args.cut_distance,
        show_leaf_counts=args.dendrogram_show_leaf_counts,
        distance_sort=args.dendrogram_distance_sort,
    )
    print(f"Dendrogram saved to '{dendrogram_path}'.")

    _analyze_inconsistency(
        matrix=matrix,
        linkage_matrix=linkage_matrix,
        depth=args.inconsistency_depth,
        top_k=max(args.inconsistency_top_k, 0),
        tail_count=max(args.inconsistency_tail, 0),
        window=max(args.inconsistency_window, 1),
        threshold=args.inconsistency_threshold,
    )

    cluster_labels = _assign_clusters(
        linkage_matrix=linkage_matrix,
        cut_distance=args.cut_distance,
        cut_maxclust=args.cut_maxclust,
    )

    if cluster_labels is None:
        print(
            "\nNo cut-distance or cut-maxclust provided; skipping cluster assignment "
            "and downstream visualisations."
        )
        return

    df_with_labels = df.copy()
    df_with_labels["Group"] = cluster_labels

    cluster_output_path = Path(args.cluster_output)
    cluster_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_labels.to_csv(cluster_output_path, index=False)
    print(f"Cluster assignments saved to '{cluster_output_path}'.")

    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print("\nTaille des groupes (nombre de formations par cluster):")
    for cluster_id, count in zip(unique_clusters, counts):
        print(f"  Groupe {cluster_id}: {count}")

    cluster_dendrogram_path = Path(args.cluster_dendrogram_path)
    cluster_title_parts = [f"{len(unique_clusters)} clusters"]
    if args.cut_distance is not None:
        cluster_title_parts.append(f"cut_distance={args.cut_distance:.2f}")
    if args.cut_maxclust is not None:
        cluster_title_parts.append(f"maxclust={args.cut_maxclust}")
    cluster_title = "Dendrogramme orienté (" + ", ".join(cluster_title_parts) + ")"
    leaf_labels = [str(idx) for idx in range(1, len(cluster_labels) + 1)]
    height = max(10.0, len(cluster_labels) * 0.12)
    # Build a consistent color map for clusters and apply to dendrogram leaves
    cluster_color_map = _build_cluster_color_map(cluster_labels)

    _plot_and_save_dendrogram(
        linkage_matrix=linkage_matrix,
        output_path=cluster_dendrogram_path,
        truncate_p=0,
        title=cluster_title,
        orientation="right",
        labels=leaf_labels,
        color_threshold=(
            args.dendrogram_color_threshold
            if args.dendrogram_color_threshold is not None
            else args.cut_distance
        ),
        cut_distance=args.cut_distance,
        show_leaf_counts=True,
        distance_sort="descending",
        figsize=(12, height),
        leaf_color_map={label: cluster_color_map[str(cluster_labels[idx])]
                        for idx, label in enumerate(leaf_labels)},
    )
    print(f"Cluster-oriented dendrogram saved to '{cluster_dendrogram_path}'.")

    palette = _build_color_palette(len(unique_clusters))

    if not args.skip_interactive:
        _create_interactive_plot(
            df_with_meta=df_with_labels,
            feature_matrix=features,
            cluster_labels=cluster_labels,
            output_path=Path(args.interactive_plot_path),
        )

    if not args.skip_stats:
        stats_df = _load_clean_features_with_groups(
            df_with_clusters=df_with_labels,
            clean_data_path=args.clean_data_path,
        )
        if stats_df is None:
            stats_df = df_with_labels
        _create_group_statistics(
            df_with_clusters=stats_df,
            palette=palette,
            output_dir=Path(args.static_plots_dir),
        )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

