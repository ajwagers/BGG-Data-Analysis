import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import gower
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_gower_distance(data: pd.DataFrame) -> np.ndarray:
    """Calculates Gower distance matrix."""
    data_for_gower = data.drop(columns=['primary_name', 'PC1', 'PC2'])

    # Manually create a boolean mask for categorical features.
    cat_features_mask = [dtype.name == 'category' for dtype in data_for_gower.dtypes]

    logger.info("Calculating Gower distance matrix... (this may take a moment)")
    distance_matrix = gower.gower_matrix(data_for_gower, cat_features=cat_features_mask)
    logger.info("Gower distance matrix calculated.")

    return distance_matrix

def perform_knn_clustering(distance_matrix: np.ndarray, data: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """Performs k-nearest neighbors clustering using Gower distance."""
    logger.info(f"Finding {n_neighbors} nearest neighbors for each game...")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='precomputed')
    knn.fit(distance_matrix)

    # Find the nearest neighbors
    distances, indices = knn.kneighbors()

    nearest_neighbor_names = []
    for idx in indices:
        neighbor_names = data.iloc[idx]['primary_name'].tolist()
        nearest_neighbor_names.append(neighbor_names[1:])  # Skip the first entry which is the game itself

    data['nearest_neighbors'] = nearest_neighbor_names
    logger.info("KNN clustering complete.")

    return data

def main():
    """Main function to run KNN clustering."""
    input_file = 'bgg_pca_output.csv'
    try:
        df_data = pd.read_csv(input_file)

        # Ensure categorical columns are properly categorized
        for col in df_data.select_dtypes(include=['object']).columns:
            if col not in ['primary_name']:
                df_data[col] = df_data[col].astype('category')

        distance_matrix = calculate_gower_distance(df_data)

        if np.any(np.isnan(distance_matrix)):
            logger.error("Distance matrix contains NaN values. Ensure all data is properly imputed.")
            return

        df_knn = perform_knn_clustering(distance_matrix, df_data, n_neighbors=5)

        # Save the processed DataFrame with nearest neighbors for later use
        output_file = 'bgg_knn_output.csv'
        df_knn.to_csv(output_file, index=False)
        logger.info(f"KNN clustering results saved to {output_file}")
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}. Please run bgg_pca_analysis.py first.")

if __name__ == "__main__":
    main()