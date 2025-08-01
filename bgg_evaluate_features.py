import pandas as pd
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Any

# --- Constants ---
RANDOM_STATE = 42

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_feature_set(filepath: Path, cluster_range: List[int]) -> Dict[str, Any]:
    """
    Calculates the best silhouette score for a given dataset over a range of cluster numbers.

    Args:
        filepath (Path): Path to the PCA output CSV file.
        cluster_range (List[int]): A list of k values to test for clustering.

    Returns:
        Dict[str, Any]: A dictionary containing the subset name, best k, and best score, or None on error.
    """
    logger.info(f"Evaluating {filepath.name}...")
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
    Finds all PCA output files, evaluates them, and reports the best feature set.
    """
    project_dir = Path('.')  # Assumes the script is run from the project root
    cluster_range_to_test = [3, 4, 5, 6, 7]

    pca_files = list(project_dir.glob('bgg_pca_output_*.csv'))
    if not pca_files:
        logger.error("No 'bgg_pca_output_*.csv' files found. Please run bgg_pca_analysis.py first.")
        return

    logger.info(f"Found {len(pca_files)} feature set files to evaluate.")
    all_results = [res for file in pca_files if (res := evaluate_feature_set(file, cluster_range_to_test)) is not None]

    if not all_results:
        logger.error("Evaluation failed for all found files.")
        return

    results_df = pd.DataFrame(all_results).sort_values(by="Best Silhouette Score", ascending=False).reset_index(drop=True)

    print("\n" + "="*80)
    print("           FEATURE SET EVALUATION RESULTS (HIGHER SCORE IS BETTER)")
    print("="*80 + "\n" + results_df.to_string() + "\n" + "="*80 + "\n")

    winner = results_df.iloc[0]
    logger.info(f"🏆 WINNER: The best feature set is '{winner['Feature Subset']}' with a silhouette score of {winner['Best Silhouette Score']:.4f} at k={winner['Best k']}.")
    logger.info(f"\nTo explore the winning cluster in detail, run bgg_visualizations.py and set the input file to:\n  input_file = 'bgg_pca_output_{winner['Feature Subset']}.csv'")

if __name__ == "__main__":
    main()
