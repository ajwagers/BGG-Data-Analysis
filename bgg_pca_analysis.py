import pandas as pd
import numpy as np
import gower
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts the primary feature from semi-colon delimited string columns."""
    # A dictionary to map original columns to new 'main' columns
    feature_map = {
        'categories': 'main_category',
        'mechanics': 'main_mechanic',
        'designers': 'main_designer',
        'artists': 'main_artist',
        'publishers': 'main_publisher',
        'families': 'main_family',
    }

    df_engineered = df.copy()
    for original_col, new_col in feature_map.items():
        if original_col in df_engineered.columns:
            logger.info(f"Extracting primary feature from '{original_col}' into '{new_col}'.")
            df_engineered[new_col] = df_engineered[original_col].str.split(';').str[0].str.strip()
            df_engineered = df_engineered.drop(columns=[original_col])
    return df_engineered

def load_and_clean_data(filepath: str, features_to_use: list) -> pd.DataFrame:
    """Loads and preprocesses the BGG data for analysis."""
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}. Please run bgg_data_ingestion.py first.")
        return None

    # --- Feature Selection ---
    existing_features = [f for f in features_to_use if f in df.columns]
    df_analysis = df[existing_features].copy()

    # --- Drop fully empty columns ---
    cols_to_drop = [col for col in df_analysis.columns if df_analysis[col].isnull().all()]
    if cols_to_drop:
        logger.warning(f"Dropping the following columns because they are completely empty: {cols_to_drop}")
        df_analysis = df_analysis.drop(columns=cols_to_drop)

    # --- Feature Engineering for Categorical Data ---
    # Note: Taking only the first item is a simplification and loses information from multi-value fields.
    df_analysis = _engineer_features(df_analysis)

    # --- Handling Missing Values ---
    numerical_cols = df_analysis.select_dtypes(include=np.number).columns
    # Exclude 'primary_name' from categorical columns to be imputed
    categorical_cols_to_impute = [col for col in df_analysis.select_dtypes(include=['object', 'category']).columns if col != 'primary_name']

    if not numerical_cols.empty:
        num_imputer = SimpleImputer(strategy='median')
        imputed_numerical_data = num_imputer.fit_transform(df_analysis[numerical_cols])
        df_analysis[numerical_cols] = pd.DataFrame(imputed_numerical_data, index=df_analysis.index, columns=numerical_cols)

    if categorical_cols_to_impute:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        imputed_categorical_data = cat_imputer.fit_transform(df_analysis[categorical_cols_to_impute])
        df_analysis[categorical_cols_to_impute] = pd.DataFrame(imputed_categorical_data, index=df_analysis.index, columns=categorical_cols_to_impute)

    # Convert all object columns (that are not primary_name) to category for Gower distance
    categorical_cols = df_analysis.select_dtypes(include=['object']).columns.drop('primary_name', errors='ignore')
    for col in categorical_cols:
        df_analysis[col] = df_analysis[col].astype('category')

    logger.info("Data cleaning and preprocessing complete.")
    logger.info(f"Shape of data for analysis: {df_analysis.shape}")
    logger.info(f"Columns for analysis: {df_analysis.columns.tolist()}")

    return df_analysis

def generate_low_correlation_subsets(
    base_data_path: str,
    subset_sizes: List[int],
    candidate_features: List[str]
) -> Dict[str, List[str]]:
    """
    Generates feature subsets by greedily selecting features with low mutual correlation.

    Args:
        base_data_path: Path to the raw, unprocessed CSV data.
        subset_sizes: A list of integers for the desired size of each subset.
        candidate_features: A list of numerical column names to consider.

    Returns:
        A dictionary where keys are subset names (e.g., "low_corr_3") and
        values are lists of selected feature names.
    """
    logger.info("--- Generating Low-Correlation Feature Subsets ---")
    try:
        df = pd.read_csv(base_data_path)
    except FileNotFoundError:
        logger.error(f"Base data file not found at {base_data_path}. Cannot generate low-correlation subsets.")
        return {}

    existing_candidates = [f for f in candidate_features if f in df.columns]
    if len(existing_candidates) < max(subset_sizes, default=0):
        logger.error(f"Not enough candidate features ({len(existing_candidates)}) exist in the data to create the largest requested subset size.")
        return {}
    
    logger.info(f"Using {len(existing_candidates)} existing features for correlation analysis.")
    
    corr_matrix = df[existing_candidates].corr().abs()
    
    s = corr_matrix.unstack()
    s = s[s.index.get_level_values(0) != s.index.get_level_values(1)]
    if s.empty:
        logger.error("Could not find any feature pairs to start correlation analysis.")
        return {}
        
    selected_features = list(s.idxmin())
    low_corr_subsets = {}

    for size in sorted(subset_sizes):
        if size < 2: continue
        
        while len(selected_features) < size:
            remaining_features = corr_matrix.index.difference(selected_features)
            if remaining_features.empty: break

            avg_correlations = corr_matrix.loc[selected_features, remaining_features].mean(axis=0)
            selected_features.append(avg_correlations.idxmin())
        
        subset_key = f"low_corr_{len(selected_features)}"
        low_corr_subsets[subset_key] = ['primary_name'] + selected_features
        logger.info(f"Generated subset '{subset_key}' with features: {selected_features}")
    return low_corr_subsets

def perform_pcoa(data: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    """Calculates Gower distance, saves it, and performs PCoA (MDS)."""
    data_for_gower = data.drop(columns=['primary_name'])
    cat_features_mask = [dtype.name == 'category' for dtype in data_for_gower.dtypes]

    logger.info("Calculating Gower distance matrix... (this may take a moment)")
    distance_matrix = gower.gower_matrix(data_for_gower, cat_features=cat_features_mask)
    logger.info("Gower distance matrix calculated.")

    # --- Save the distance matrix for use in other scripts ---
    matrix_output_file = f'gower_distance_matrix_{subset_name}.npy'
    np.save(matrix_output_file, distance_matrix)
    logger.info(f"Gower distance matrix saved to {matrix_output_file}")

    n_components = 6
    logger.info(f"Performing Principal Coordinate Analysis (PCoA) for {n_components} components...")
    # n_init=4 (default) is more robust than n_init=1 but slower.
    pcoa = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, n_init=4, max_iter=300)
    coords = pcoa.fit_transform(distance_matrix)
    logger.info(f"PCoA stress for {n_components} components (lower is better): {pcoa.stress_:.4f}")

    for i in range(n_components):
        data[f'PC{i+1}'] = coords[:, i]
    logger.info("PCoA complete.")

    return data

def perform_full_analysis_pipeline(filepath: str, features_to_use: list, subset_name: str) -> pd.DataFrame:
    """Loads, cleans the data and performs PCoA."""
    df_cleaned = load_and_clean_data(filepath, features_to_use)
    if df_cleaned is not None:
        df_pcoa = perform_pcoa(df_cleaned, subset_name)
        return df_pcoa
    else:
        return None

def main():
    """Main function to run the analysis."""
    csv_file = 'bgg_top_games_updated.csv' # This should be the output of your ingestion script

    # --- Define Feature Subsets for Experimentation ---
    # Each subset is a hypothesis about what makes games similar.
    # 'primary_name' is always included as it's the identifier.
    FEATURE_SUBSETS = {
        "core_gameplay": [
            'primary_name', 'year_published', 'min_players', 'max_players',
            'min_playing_time', 'max_playing_time', 'min_age', 'average_weight',
            'categories', 'mechanics'
        ],
        "community_ratings": [
            'primary_name', 'average_rating', 'bayes_average', 'stddev',
            'median', 'users_rated', 'num_weights', 'average_weight'
        ],
        "market_popularity": [
            'primary_name', 'owned', 'wishing', 'trading', 'wanting',
            'num_comments', 'users_rated', 'bgg_rank'
        ],
        "full_set": [
            'primary_name', 'year_published', 'min_players', 'max_players',
            'min_playing_time', 'max_playing_time', 'min_age', 'average_rating',
            'bayes_average', 'average_weight', 'bgg_rank', 'categories', 'mechanics', 'families',
            'owned', 'users_rated', 'trading', 'wanting', 'wishing', 'num_comments',
            'num_weights', 'stddev', 'median', 'designers', 'artists', 'publishers'
        ]
    }

    # --- Generate and add low-correlation subsets ---
    # A pool of numerical features that are likely to have interesting relationships
    CANDIDATE_NUMERICAL_FEATURES = [
        'year_published', 'min_players', 'max_players', 'min_playing_time',
        'max_playing_time', 'min_age', 'average_weight', 'average_rating',
        'bayes_average', 'stddev', 'users_rated', 'num_weights', 'owned',
        'wishing', 'trading', 'wanting', 'num_comments', 'bgg_rank'
    ]
    
    low_corr_subsets = generate_low_correlation_subsets(
        base_data_path=csv_file,
        subset_sizes=list(range(3, 8)), # For subsets of size 3, 4, 5, 6, 7
        candidate_features=CANDIDATE_NUMERICAL_FEATURES
    )
    
    FEATURE_SUBSETS.update(low_corr_subsets)

    # --- Run Analysis for Each Subset ---
    for subset_name, feature_list in FEATURE_SUBSETS.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING ANALYSIS FOR FEATURE SUBSET: '{subset_name}'")
        logger.info(f"{'='*80}\n")

        df_pcoa = perform_full_analysis_pipeline(csv_file, feature_list, subset_name)

        if df_pcoa is not None:
            output_file = f'bgg_pca_output_{subset_name}.csv'
            df_pcoa.to_csv(output_file, index=False)
            logger.info(f"Processed data for '{subset_name}' saved to {output_file}")

if __name__ == "__main__":
    main()