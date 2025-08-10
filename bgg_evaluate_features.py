import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Constants ---
RANDOM_STATE = 42

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_silhouette_analysis(data: pd.DataFrame, cluster_range: List[int], output_dir: Path, subset_name: str):
    """
    Performs silhouette analysis for KMeans clustering on a range of n_clusters
    and saves the plots to files.
    """
    X = data[['PC1', 'PC2']].values

    for n_clusters in cluster_range:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        logger.info(f"  For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg:.4f}")

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))
            y_lower = y_upper + 10

        ax1.set_title("Silhouette plot for the various clusters")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker=f'${i + 1}$', alpha=1, s=50, edgecolor='k')

        ax2.set_title("Visualization of the clustered data")
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")

        plt.suptitle((f"Silhouette analysis for '{subset_name}' with n_clusters = {n_clusters}"),
                     fontsize=14, fontweight='bold')

        silhouette_plot_filename = output_dir / f'silhouette_analysis_{n_clusters}_clusters.png'
        try:
            plt.savefig(silhouette_plot_filename, bbox_inches='tight')
            logger.info(f"  -> Silhouette plot saved to {silhouette_plot_filename}")
        except Exception as e:
            logger.error(f"  -> Failed to save plot {silhouette_plot_filename}: {e}")
        finally:
            plt.close(fig)  # Close the figure to free up memory

def evaluate_feature_set(filepath: Path, cluster_range: List[int]) -> Dict[str, Any]:
    """
    Calculates the best silhouette score for a given dataset over a range of cluster numbers.

    Args:
        filepath (Path): Path to the PCA output CSV file.
        cluster_range (List[int]): A list of k values to test for clustering.

    Returns:
        Dict[str, Any]: A dictionary containing the subset name, best k, and best score, or None on error.
    """
    logger.info(f"--- Evaluating Scores for '{filepath.stem.replace('bgg_pca_output_', '')}' ---")
    try:
        df = pd.read_csv(filepath)
        # Ensure the required columns exist
        if 'PC1' not in df.columns or 'PC2' not in df.columns:
            raise KeyError("CSV file must contain 'PC1' and 'PC2' columns.")
        X = df[['PC1', 'PC2']].values
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Could not process {filepath.name}: {e}")
        return None

    best_score = -1
    best_k = -1

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k

    # Extract subset name from filename, e.g., "bgg_pca_output_core_gameplay.csv" -> "core_gameplay"
    subset_name = filepath.stem.replace('bgg_pca_output_', '')

    result = {
        "Feature Subset": subset_name,
        "Best k": best_k,
        "Best Silhouette Score": best_score
    }
    logger.info(f"  -> Best result for '{subset_name}': k={best_k} with score {best_score:.4f}")
    return result

def main():
    """
    Finds all PCA output files, generates silhouette plots for each,
    evaluates them, and reports the best feature set.
    """
    project_dir = Path('.')
    cluster_range_to_test = [3, 4, 5, 6, 7]

    pca_files = list(project_dir.glob('bgg_pca_output_*.csv'))
    if not pca_files:
        logger.error("No 'bgg_pca_output_*.csv' files found. Please run bgg_pca_analysis.py first.")
        return
    
    # --- Part 1: Generate Silhouette Plots for all subsets ---
    EVALUATION_PLOTS_DIR = Path("evaluation_plots")
    EVALUATION_PLOTS_DIR.mkdir(exist_ok=True)
    logger.info(f"Silhouette plots will be saved to '{EVALUATION_PLOTS_DIR}/'")
    
    logger.info("\n" + "="*80)
    logger.info("         GENERATING SILHOUETTE PLOTS FOR EACH FEATURE SUBSET")
    logger.info("="*80)
    for file in pca_files:
        subset_name = file.stem.replace('bgg_pca_output_', '')
        logger.info(f"\n--- Processing plots for subset: '{subset_name}' ---")
        
        subset_plot_dir = EVALUATION_PLOTS_DIR / subset_name
        subset_plot_dir.mkdir(exist_ok=True)
        
        try:
            df = pd.read_csv(file)
            if 'PC1' not in df.columns or 'PC2' not in df.columns:
                raise KeyError("CSV file must contain 'PC1' and 'PC2' columns.")
            plot_silhouette_analysis(df, cluster_range_to_test, subset_plot_dir, subset_name)
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Could not generate plots for {file.name}: {e}")
            continue

    # --- Part 2: Evaluate scores and find the best subset ---
    logger.info("\n" + "="*80)
    logger.info("           EVALUATING SCORES TO FIND BEST FEATURE SUBSET")
    logger.info("="*80)
    all_results = [res for file in pca_files if (res := evaluate_feature_set(file, cluster_range_to_test)) is not None]

    if not all_results:
        logger.error("Evaluation failed for all found files.")
        return

    # --- Part 3: Summarize and save results ---
    if all_results:
        results_df = pd.DataFrame(all_results).sort_values(by="Best Silhouette Score", ascending=False).reset_index(drop=True)
        output_csv = "feature_set_evaluation_results.csv"
        results_df.to_csv(output_csv, index=False)
        logger.info(f"Evaluation results saved to {output_csv}")

        print("\n" + "="*80)
        print("           FEATURE SET EVALUATION RESULTS (HIGHER SCORE IS BETTER)")
        print("="*80 + "\n" + results_df.to_string() + "\n" + "="*80 + "\n")

        winner = results_df.iloc[0]
        logger.info(f"🏆 WINNER: The best feature set is '{winner['Feature Subset']}' with a silhouette score of {winner['Best Silhouette Score']:.4f} at k={winner['Best k']}.")
        logger.info(f"\nTo explore the winning cluster in detail, run bgg_visualizations.py. It will automatically use this result.")

if __name__ == "__main__":
    main()
