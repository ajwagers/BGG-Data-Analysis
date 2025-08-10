import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import List
import textwrap

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
OUTPUT_DIR = Path("visualizations")
RANDOM_STATE = 42

def plot_correlation_heatmap(data: pd.DataFrame, output_dir: Path):
    """Plots correlation heatmap of numerical features."""
    logger.info("Generating Correlation Heatmap...")

    # Select only numerical columns
    numeric_data = data.select_dtypes(include=np.number)
    correlation_matrix = numeric_data.corr()

    # For a large number of features, annotations become unreadable.
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numerical Features')

    heatmap_filename = output_dir / 'correlation_heatmap.png'
    plt.savefig(heatmap_filename, bbox_inches='tight')
    logger.info(f"Correlation heatmap saved to {heatmap_filename}")
    plt.show()

def plot_pc_relationships(data: pd.DataFrame, output_dir: Path):
    """
    Generates scatter plots and a pair plot to explore relationships
    between Principal Components and original features.
    """
    logger.info("Generating plots for PC relationships...")

    # --- Plot 1: PC1 vs. BGG Rank ---
    if 'bgg_rank' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='PC1', y='bgg_rank', alpha=0.5)
        plt.title('Relationship between PC1 and BGG Rank')
        plt.xlabel('Principal Component 1')
        plt.ylabel('BGG Rank (Lower is Better)')
        # Invert y-axis because a lower rank is better, making an upward trend more intuitive.
        plt.gca().invert_yaxis()
        plt.grid(True)
        pc1_rank_filename = output_dir / 'pc1_vs_bgg_rank.png'
        plt.savefig(pc1_rank_filename, bbox_inches='tight')
        logger.info(f"PC1 vs. Rank plot saved to {pc1_rank_filename}")
        plt.show()

    # --- Plot 2: PC1 vs. Average Weight ---
    if 'average_weight' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='PC1', y='average_weight', alpha=0.5)
        plt.title('Relationship between PC1 and Average Weight (Complexity)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Average Weight')
        plt.grid(True)
        pc1_weight_filename = output_dir / 'pc1_vs_average_weight.png'
        plt.savefig(pc1_weight_filename, bbox_inches='tight')
        logger.info(f"PC1 vs. Weight plot saved to {pc1_weight_filename}")
        plt.show()

    # --- Plot 3: Pair Plot ---
    logger.info("Generating pair plot (this might take a moment)...")
    # Use all numerical columns from the winning subset for the pair plot.
    # This can be slow and memory-intensive if the subset has many features.
    pairplot_cols = data.select_dtypes(include=np.number).columns.tolist()
    
    pair_plot = sns.pairplot(data[pairplot_cols], plot_kws={'alpha': 0.3})
    pair_plot.fig.suptitle('Pair Plot of Key Features and Principal Components', y=1.02, fontsize=16)
    pairplot_filename = output_dir / 'pairplot_features_vs_pcs.png'
    plt.savefig(pairplot_filename, bbox_inches='tight')
    logger.info(f"Pair plot saved to {pairplot_filename}")
    plt.show()

def find_central_games(data: pd.DataFrame, kmeans_model: KMeans) -> pd.DataFrame:
    """
    Finds the most central game for each cluster.
    The central game is the one closest to the cluster's centroid in the PC space.

    Args:
        data (pd.DataFrame): The dataframe containing the data and cluster assignments.
        kmeans_model (KMeans): The fitted KMeans model object.

    Returns:
        pd.DataFrame: A dataframe with the central game for each cluster.
    """
    centroids = kmeans_model.cluster_centers_
    central_games_info = []

    for i, centroid in enumerate(centroids):
        # Filter data for the current cluster
        cluster_data = data[data['cluster'] == i].copy()

        # Calculate Euclidean distance from each point in the cluster to the centroid
        distances = euclidean_distances(cluster_data[['PC1', 'PC2']].values, [centroid])
        cluster_data['distance_to_centroid'] = distances

        # Find the index of the game with the minimum distance
        closest_game_idx = cluster_data['distance_to_centroid'].idxmin()

        # Get the details of that game
        central_game_series = data.loc[closest_game_idx]
        central_games_info.append({
            "Cluster": i + 1,
            "Central Game": central_game_series['primary_name'],
            "Distance to Centroid": cluster_data.loc[closest_game_idx, 'distance_to_centroid']
        })
    return pd.DataFrame(central_games_info)

def analyze_and_plot_clusters(data: pd.DataFrame, n_clusters: int, output_dir: Path):
    """
    Performs KMeans clustering with the chosen k, plots the results,
    and prints a descriptive analysis of each cluster.
    """
    logger.info(f"\n--- Analyzing and Plotting Final Clusters (k={n_clusters}) ---")
    X = data[['PC1', 'PC2']].values

    # --- Perform Clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    data['cluster'] = kmeans.fit_predict(X)

    # --- Analyze Clusters ---
    cluster_summaries = []
    for i in range(n_clusters):
        cluster_data = data[data['cluster'] == i]
        summary = {
            "Cluster": i + 1,
            "Size": len(cluster_data),
            "Avg. Weight": cluster_data['average_weight'].mean(),
            "Avg. Rating": cluster_data['average_rating'].mean(),
            "Avg. Year": cluster_data['year_published'].mean(),
            "Top Categories": cluster_data['main_category'].value_counts().nlargest(5).index.tolist(),
            "Top Mechanics": cluster_data['main_mechanic'].value_counts().nlargest(5).index.tolist(),
        }
        cluster_summaries.append(summary)

    summary_df = pd.DataFrame(cluster_summaries)
    print("\n--- Cluster Analysis Summary ---")
    print(summary_df.to_string(index=False))

    # --- Find and Display Central Games ---
    central_games_df = find_central_games(data, kmeans)
    print("\n--- Most Representative (Central) Game of Each Cluster ---")
    print(central_games_df.to_string(index=False))

    # --- Generate Detailed Plot ---
    plt.figure(figsize=(16, 10))
    ax = sns.scatterplot(
        x='PC1', y='PC2',
        hue='cluster',
        palette=sns.color_palette("hsv", n_clusters),
        data=data,
        legend='full',
        alpha=0.8,
        s=50
    )

    # Create descriptive labels for the legend
    legend_labels = []
    for i, row in summary_df.iterrows():
        label_text = (
            f"Cluster {row['Cluster']} (n={row['Size']})\n"
            f"Avg Weight: {row['Avg. Weight']:.2f}\n"
            f"Category: {textwrap.fill(', '.join(row['Top Categories'][:2]), 25)}\n"
            f"Mechanic: {textwrap.fill(', '.join(row['Top Mechanics'][:2]), 25)}"
        )
        legend_labels.append(label_text)

    # Place legend outside the plot
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, legend_labels, title="Cluster Profiles", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title(f'Final Game Clusters (k={n_clusters}) on PCoA Space', fontsize=16)
    plt.xlabel('Principal Component 1 (Complexity/Niche Axis)')
    plt.ylabel('Principal Component 2')
    plt.grid(True)

    final_plot_filename = output_dir / f'final_{n_clusters}_clusters_plot.png'
    plt.savefig(final_plot_filename, bbox_inches='tight')
    logger.info(f"Final cluster plot saved to {final_plot_filename}")
    plt.show()

def main():
    """Main function to run visualizations."""
    try:
        # 1. Read evaluation results to find the best feature set and optimal k
        evaluation_results_file = 'feature_set_evaluation_results.csv'
        results_df = pd.read_csv(evaluation_results_file)
        winner = results_df.iloc[0]
        winner_subset = winner['Feature Subset']
        optimal_k = int(winner['Best k'])
        logger.info(f"🏆 Automatically selected winning feature set: '{winner_subset}' with optimal k={optimal_k}")

        # 2. Load the data for the winning subset
        input_file = f'bgg_pca_output_{winner_subset}.csv'
        df_data = pd.read_csv(input_file)
        logger.info(f"Loading data from the winning subset file: {input_file}")

        # Ensure categorical columns are properly categorized
        for col in df_data.select_dtypes(include=['object']).columns:
            if col not in ['primary_name']:
                df_data[col] = df_data[col].astype('category')

        # 3. Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)
        logger.info(f"Visualizations will be saved to '{OUTPUT_DIR}/'")
        
        logger.info(f"\n--- Generating Deep-Dive Visualizations for '{winner_subset}' ---")
        logger.info(f"Note: Silhouette plots for all subsets are in the 'evaluation_plots/' directory.")

        # 4. Plot a correlation heatmap of all numerical features from the original dataset
        raw_data_file = 'bgg_top_games_updated.csv'
        try:
            df_raw = pd.read_csv(raw_data_file)
            plot_correlation_heatmap(df_raw, OUTPUT_DIR)
        except FileNotFoundError:
            logger.warning(f"Raw data file '{raw_data_file}' not found. Skipping full correlation heatmap.")
            logger.warning("Generating heatmap for the winning subset only.")
            plot_correlation_heatmap(df_data, OUTPUT_DIR)

        # 5. Plot relationships between PCs and key features for the winning subset
        plot_pc_relationships(df_data, OUTPUT_DIR)

        # 6. Perform final clustering and analysis with the automatically determined k.
        logger.info(f"\nProceeding with final analysis using automatically determined k={optimal_k}...")
        analyze_and_plot_clusters(df_data, n_clusters=optimal_k, output_dir=OUTPUT_DIR)

    except FileNotFoundError as e:
        logger.error(f"Could not find a required file: {e.filename}. Please ensure you have run bgg_pca_analysis.py and bgg_evaluate_features.py first.")

if __name__ == "__main__":
    main()