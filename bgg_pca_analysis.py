import pandas as pd
import numpy as np
import gower
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
from sklearn.impute import SimpleImputer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Loads and preprocesses the BGG data for analysis."""
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}. Please run bgg_data_ingestion.py first.")
        return None

    # --- Feature Selection ---
    # Select features that are relevant for finding intrinsic properties of games
    features = [
        'primary_name', 'year_published', 'min_players', 'max_players',
        'min_playing_time', 'max_playing_time', 'min_age', 'average_rating',
        'bayes_average', 'average_weight', 'bgg_rank', 'categories', 'mechanics'
    ]
    # Ensure we only select columns that actually exist in the CSV
    existing_features = [f for f in features if f in df.columns]
    df_analysis = df[existing_features].copy()

    # --- Drop fully empty columns ---
    # This can happen if the API returns no data for a field for any game in the set.
    # This also prevents the imputer from creating a shape mismatch.
    cols_to_drop = [col for col in df_analysis.columns if df_analysis[col].isnull().all()]
    if cols_to_drop:
        logger.warning(f"Dropping the following columns because they are completely empty: {cols_to_drop}")
        df_analysis = df_analysis.drop(columns=cols_to_drop)

    # --- Feature Engineering for Categorical Data ---
    # Gower's distance works best with single categories, not long strings of them.
    # We'll extract the primary category and mechanic as a simplification.
    if 'categories' in df_analysis.columns:
        df_analysis['main_category'] = df_analysis['categories'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['categories'])

    if 'mechanics' in df_analysis.columns:
        df_analysis['main_mechanic'] = df_analysis['mechanics'].str.split(';').str[0].str.strip()
        df_analysis = df_analysis.drop(columns=['mechanics'])

    # --- Handling Missing Values ---
    # Identify numerical and categorical columns for imputation
    numerical_cols = df_analysis.select_dtypes(include=np.number).columns
    categorical_cols = df_analysis.select_dtypes(include=['object', 'category']).columns.drop('primary_name', errors='ignore')

    # Impute numerical columns with the median
    if not numerical_cols.empty:
        num_imputer = SimpleImputer(strategy='median')
        # Reconstruct DataFrame to preserve column names and index, preventing assignment errors
        imputed_numerical_data = num_imputer.fit_transform(df_analysis[numerical_cols])
        df_analysis[numerical_cols] = pd.DataFrame(imputed_numerical_data, index=df_analysis.index, columns=numerical_cols)

    # Impute categorical columns with the most frequent value
    if not categorical_cols.empty:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        # Reconstruct DataFrame to preserve column names and index
        imputed_categorical_data = cat_imputer.fit_transform(df_analysis[categorical_cols])
        df_analysis[categorical_cols] = pd.DataFrame(imputed_categorical_data, index=df_analysis.index, columns=categorical_cols)

    # Convert categorical columns to 'category' dtype for Gower
    for col in categorical_cols:
        df_analysis[col] = df_analysis[col].astype('category')

    logger.info("Data cleaning and preprocessing complete.")
    logger.info(f"Shape of data for analysis: {df_analysis.shape}")
    logger.info(f"Columns for analysis: {df_analysis.columns.tolist()}")
    
    return df_analysis

def perform_pcoa(data: pd.DataFrame):
    """Calculates Gower distance and performs PCoA (MDS)."""
    # We don't want the game name in the distance calculation
    data_for_gower = data.drop(columns=['primary_name'])

    # Manually create a boolean mask for categorical features.
    # This is necessary because numpy.issubdtype, used by gower internally,
    # can't handle pandas' CategoricalDtype.
    cat_features_mask = [dtype.name == 'category' for dtype in data_for_gower.dtypes]

    # --- Gower Distance ---
    logger.info("Calculating Gower distance matrix... (this may take a moment)")
    distance_matrix = gower.gower_matrix(data_for_gower, cat_features=cat_features_mask)
    logger.info("Gower distance matrix calculated.")

    # --- PCoA (using MDS) ---
    # PCoA is equivalent to classical/metric MDS
    logger.info("Performing Principal Coordinate Analysis (PCoA)...")
    pcoa = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=300)
    coords = pcoa.fit_transform(distance_matrix)
    
    data['PC1'] = coords[:, 0]
    data['PC2'] = coords[:, 1]
    logger.info("PCoA complete.")
    
    return data

def analyze_principal_components(data: pd.DataFrame):
    """Analyzes and visualizes the principal components."""
    
    # --- Analyze PC1 ---
    # Correlate numerical features with the first principal component
    numerical_features = data.select_dtypes(include=np.number).columns.drop(['PC1', 'PC2'])
    correlations = data[numerical_features].corrwith(data['PC1'])
    
    print("\n--- Analysis of Principal Component 1 (PC1) ---")
    print("PC1 represents the primary axis of variation in the dataset.")
    print("\nCorrelation of Numerical Features with PC1:")
    print(correlations.sort_values(ascending=False))

    # --- Visualization ---
    logger.info("Generating plot...")
    plt.figure(figsize=(12, 8))

    # Dynamically determine the columns for hue and size to avoid errors if they were dropped
    hue_col = 'average_weight' if 'average_weight' in data.columns else None
    size_col = 'average_rating' if 'average_rating' in data.columns else None

    # If 'average_rating' was dropped, use 'bayes_average' as a fallback for sizing points
    if size_col is None and 'bayes_average' in data.columns:
        size_col = 'bayes_average'
        logger.info("Using 'bayes_average' for plot point size as 'average_rating' is not available.")

    sns.scatterplot(
        x='PC1', y='PC2', data=data,
        hue=hue_col, palette='viridis',
        size=size_col, sizes=(20, 200), alpha=0.7
    )
    plt.title('PCoA of BGG Hot Games (based on Gower Distance)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    
    plot_filename = 'bgg_pcoa_plot.png'
    plt.savefig(plot_filename)
    logger.info(f"Plot saved to {plot_filename}")
    plt.show()

def main():
    """Main function to run the analysis."""
    csv_file = 'bgg_hot_games.csv'
    df_cleaned = load_and_clean_data(csv_file)
    
    if df_cleaned is not None:
        df_pcoa = perform_pcoa(df_cleaned)
        analyze_principal_components(df_pcoa)

if __name__ == "__main__":
    main()