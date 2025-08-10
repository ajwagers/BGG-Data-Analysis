import pandas as pd
import numpy as np
import hdbscan
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_hdbscan_clustering(distance_matrix: np.ndarray, data: pd.DataFrame, min_cluster_size: int = 10) -> pd.DataFrame:
    """
    Performs HDBSCAN clustering on the Gower distance matrix to find natural game clusters.
    
    Args:
        distance_matrix: A square matrix of distances between all games.
        data: The DataFrame containing game info, must include 'primary_name'.
        min_cluster_size: The minimum number of games required to form a cluster.
    
    Returns:
        The input DataFrame with an added 'cluster_label' column. Noise points are labeled -1.
    """
    logger.info(f"Performing HDBSCAN clustering with min_cluster_size={min_cluster_size}...")
    
    # HDBSCAN's internal Cython code expects a C double (np.float64) matrix.
    # The Gower matrix is often float32, so we ensure the correct dtype here.
    distance_matrix_double = distance_matrix.astype(np.float64)

    # HDBSCAN is great for this because it can work directly on a distance matrix
    # and doesn't require us to specify the number of clusters beforehand.
    clusterer = hdbscan.HDBSCAN(metric='precomputed', 
                                min_cluster_size=min_cluster_size,
                                allow_single_cluster=True)
    
    clusterer.fit(distance_matrix_double)
    
    data['cluster_label'] = clusterer.labels_
    
    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_noise = np.sum(clusterer.labels_ == -1)
    
    logger.info(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    logger.info("Clustering complete.")

    return data

def main():
    """Main function to run HDBSCAN clustering for each feature subset."""
    # In a larger project, this might be moved to a shared config file.
    FEATURE_SUBSETS = [
        "core_gameplay",
        "community_ratings",
        "market_popularity",
        "full_set"
    ]

    for subset_name in FEATURE_SUBSETS:
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING HDBSCAN CLUSTERING FOR FEATURE SUBSET: '{subset_name}'")
        logger.info(f"{'='*80}\n")

        pca_input_file = f'bgg_pca_output_{subset_name}.csv'
        matrix_input_file = f'gower_distance_matrix_{subset_name}.npy'

        try:
            df_data = pd.read_csv(pca_input_file)
            distance_matrix = np.load(matrix_input_file)
            logger.info(f"Loaded data from {pca_input_file} and distance matrix from {matrix_input_file}")
        except FileNotFoundError as e:
            logger.error(f"Could not find input file: {e.filename}. Please run bgg_pca_analysis.py first.")
            continue  # Skip to the next subset

        if np.any(np.isnan(distance_matrix)):
            logger.error(f"Distance matrix for '{subset_name}' contains NaN values. Aborting this subset.")
            continue
        
        # Find clusters. min_cluster_size is a key parameter to tune.
        df_clustered = perform_hdbscan_clustering(distance_matrix, df_data, min_cluster_size=10)

        # Save the processed DataFrame with cluster labels for later use
        output_file = f'bgg_hdbscan_output_{subset_name}.csv'
        try:
            df_clustered.to_csv(output_file, index=False)
            logger.info(f"HDBSCAN clustering results for '{subset_name}' saved to {output_file}")
        except Exception as e:
            logger.error(f"Could not save output file {output_file}. Error: {e}")


if __name__ == "__main__":
    main()