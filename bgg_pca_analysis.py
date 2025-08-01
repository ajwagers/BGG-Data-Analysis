import pandas as pd
import numpy as np
import gower
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
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
    if 'categories' in df_analysis.columns:
        df_analysis['main_category'] = df_analysis['categories'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['categories'])

    if 'mechanics' in df_analysis.columns:
        df_analysis['main_mechanic'] = df_analysis['mechanics'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['mechanics'])

    if 'designers' in df_analysis.columns:
        df_analysis['main_designer'] = df_analysis['designers'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['designers'])

    if 'artists' in df_analysis.columns:
        df_analysis['main_artist'] = df_analysis['artists'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['artists'])

    if 'publishers' in df_analysis.columns:
        df_analysis['main_publisher'] = df_analysis['publishers'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['publishers'])

    if 'families' in df_analysis.columns:
        df_analysis['main_family'] = df_analysis['families'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['families'])

    # --- Handling Missing Values ---
    numerical_cols = df_analysis.select_dtypes(include=np.number).columns
    categorical_cols = df_analysis.select_dtypes(include=['object', 'category']).columns.drop('primary_name', errors='ignore')

    if not numerical_cols.empty:
        num_imputer = SimpleImputer(strategy='median')
        imputed_numerical_data = num_imputer.fit_transform(df_analysis[numerical_cols])
        df_analysis[numerical_cols] = pd.DataFrame(imputed_numerical_data, index=df_analysis.index, columns=numerical_cols)

    if not categorical_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        imputed_categorical_data = cat_imputer.fit_transform(df_analysis[categorical_cols])
        df_analysis[categorical_cols] = pd.DataFrame(imputed_categorical_data, index=df_analysis.index, columns=categorical_cols)

    for col in categorical_cols:
        df_analysis[col] = df_analysis[col].astype('category')

    logger.info("Data cleaning and preprocessing complete.")
    logger.info(f"Shape of data for analysis: {df_analysis.shape}")
    logger.info(f"Columns for analysis: {df_analysis.columns.tolist()}")

    return df_analysis

def perform_pcoa(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates Gower distance and performs PCoA (MDS)."""
    data_for_gower = data.drop(columns=['primary_name'])
    cat_features_mask = [dtype.name == 'category' for dtype in data_for_gower.dtypes]

    logger.info("Calculating Gower distance matrix... (this may take a moment)")
    distance_matrix = gower.gower_matrix(data_for_gower, cat_features=cat_features_mask)
    logger.info("Gower distance matrix calculated.")

    n_components = 6
    logger.info(f"Performing Principal Coordinate Analysis (PCoA) for {n_components} components...")
    pcoa = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=300)
    coords = pcoa.fit_transform(distance_matrix)
    logger.info(f"PCoA stress for {n_components} components (lower is better): {pcoa.stress_:.4f}")

    for i in range(n_components):
        data[f'PC{i+1}'] = coords[:, i]
    logger.info("PCoA complete.")

    return data

def perform_full_analysis_pipeline(filepath: str, features_to_use: list) -> pd.DataFrame:
    """Loads, cleans the data and performs PCoA."""
    df_cleaned = load_and_clean_data(filepath, features_to_use)
    if df_cleaned is not None:
        df_pcoa = perform_pcoa(df_cleaned)
        return df_pcoa
    else:
        return None

def main():
    """Main function to run the analysis."""
    csv_file = 'bgg_top_games_updated.csv'

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

    # --- Run Analysis for Each Subset ---
    for subset_name, feature_list in FEATURE_SUBSETS.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING ANALYSIS FOR FEATURE SUBSET: '{subset_name}'")
        logger.info(f"{'='*80}\n")

        df_pcoa = perform_full_analysis_pipeline(csv_file, feature_list)

        if df_pcoa is not None:
            output_file = f'bgg_pca_output_{subset_name}.csv'
            df_pcoa.to_csv(output_file, index=False)
            logger.info(f"Processed data for '{subset_name}' saved to {output_file}")

if __name__ == "__main__":
    main()