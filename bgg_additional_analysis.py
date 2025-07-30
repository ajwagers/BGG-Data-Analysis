import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_principal_components(data: pd.DataFrame):
    """Analyzes and visualizes the principal components."""

    # --- Analyze PC1 ---
    numerical_features = data.select_dtypes(include=np.number).columns.drop(['PC1', 'PC2'])
    correlations = data[numerical_features].corrwith(data['PC1'])

    print("\n--- Analysis of Principal Component 1 (PC1) ---")
    print("PC1 represents the primary axis of variation in the dataset.")
    print("\nCorrelation of Numerical Features with PC1:")
    print(correlations.sort_values(ascending=False))

    # --- Analyze Categorical Features vs PC1 ---
    print("\n--- Analysis of Categorical Features vs PC1 ---")

    if 'main_category' in data.columns:
        print("\nAverage PC1 Score by Main Category (Top 10 Highest/Lowest):")
        category_pc1 = data.groupby('main_category')['PC1'].mean().sort_values(ascending=False)
        print("--- HIGHEST PC1 (often more complex/niche) ---")
        print(category_pc1.head(10))
        print("\n--- LOWEST PC1 (often more accessible/mainstream) ---")
        print(category_pc1.tail(10))

    if 'main_mechanic' in data.columns:
        print("\nAverage PC1 Score by Main Mechanic (Top 10 Highest/Lowest):")
        mechanic_pc1 = data.groupby('main_mechanic')['PC1'].mean().sort_values(ascending=False)
        print("--- HIGHEST PC1 (often more complex/niche) ---")
        print(mechanic_pc1.head(10))
        print("\n--- LOWEST PC1 (often more accessible/mainstream) ---")
        print(mechanic_pc1.tail(10))

    # --- Visualization ---
    logger.info("Generating plot...")
    plt.figure(figsize=(12, 8))

    hue_col = 'average_weight' if 'average_weight' in data.columns else None
    size_col = 'average_rating' if 'average_rating' in data.columns else None

    if size_col is None and 'bayes_average' in data.columns:
        size_col = 'bayes_average'
        logger.info("Using 'bayes_average' for plot point size as 'average_rating' is not available.")

    sns.scatterplot(
        x='PC1', y='PC2', data=data,
        hue=hue_col, palette='viridis',
        size=size_col, sizes=(20, 200), alpha=0.7
    )
    plt.title('PCoA of Top 5000 BGG Games (based on Gower Distance)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    plot_filename = 'bgg_top_5000_pcoa_plot.png'
    plt.savefig(plot_filename)
    logger.info(f"Plot saved to {plot_filename}")
    plt.show()

def additional_analysis_and_visualizations(filepath: str):
    """Performs additional analysis and visualizations."""
    try:
        data = pd.read_csv(filepath)
        analyze_principal_components(data)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}. Please run bgg_pca_analysis.py first.")

def main():
    """Main function to run the analysis."""
    csv_file = 'bgg_pca_output.csv'
    additional_analysis_and_visualizations(csv_file)

if __name__ == "__main__":
    main()