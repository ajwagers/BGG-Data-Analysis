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

def plot_silhouette_analysis(data: pd.DataFrame, cluster_range: List[int], output_dir: Path):
    """
    Performs silhouette analysis for KMeans clustering on a range of n_clusters.
    This visualization is adapted from the scikit-learn documentation to provide
    a more informative plot for selecting the optimal number of clusters.
    """
    X = data[['PC1', 'PC2']].values

    for n_clusters in cluster_range:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        ax1.set_xlim([-0.2, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        silhouette_avg = silhouette_score(X, cluster_labels)
        logger.info(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg:.4f}")

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette plot for the various clusters")
        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker=f'${i+1}$', alpha=1, s=50, edgecolor='k')

        ax2.set_title("Visualization of the clustered data")
        ax2.set_xlabel("Principal Component 1")
        ax2.set_ylabel("Principal Component 2")

        plt.suptitle((f"Silhouette analysis for KMeans clustering with n_clusters = {n_clusters}"),
                     fontsize=14, fontweight='bold')

        silhouette_plot_filename = output_dir / f'silhouette_analysis_{n_clusters}_clusters.png'
        plt.savefig(silhouette_plot_filename, bbox_inches='tight')
        logger.info(f"Silhouette plot saved to {silhouette_plot_filename}")

    plt.show()

def plot_correlation_heatmap(data: pd.DataFrame, output_dir: Path):
    """Plots correlation heatmap of numerical features."""
    logger.info("Generating Correlation Heatmap...")

    # Select only numerical columns
    numeric_data = data.select_dtypes(include=np.number)
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
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
    pairplot_cols = ['PC1', 'PC2', 'bgg_rank', 'average_weight', 'average_rating', 'year_published']
    pairplot_cols_exist = [col for col in pairplot_cols if col in data.columns]
    
    pair_plot = sns.pairplot(data[pairplot_cols_exist], plot_kws={'alpha': 0.3})
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
    # --- CHOOSE THE INPUT FILE TO VISUALIZE ---
    # After running the new bgg_pca_analysis.py, you will have multiple output files.
    # Change this variable to point to the file you want to analyze.
    # Example files: 'bgg_pca_output_core_gameplay.csv', 'bgg_pca_output_community_ratings.csv', etc.
    input_file = 'bgg_pca_output_full_set.csv'
    try:
        df_data = pd.read_csv(input_file)

        # Ensure categorical columns are properly categorized
        for col in df_data.select_dtypes(include=['object']).columns:
            if col not in ['primary_name']:
                df_data[col] = df_data[col].astype('category')

        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)
        logger.info(f"Visualizations will be saved to '{OUTPUT_DIR}/'")

        # 1. Run silhouette analysis to help determine the optimal number of clusters
        cluster_range_to_test = [3, 4, 5, 6, 7]
        logger.info(f"Performing silhouette analysis for n_clusters in {cluster_range_to_test}...")
        plot_silhouette_analysis(df_data, cluster_range_to_test, OUTPUT_DIR)

        # 2. Plot a correlation heatmap of numerical features
        plot_correlation_heatmap(df_data.select_dtypes(include=np.number), OUTPUT_DIR)

        # 3. Plot relationships between PCs and key features
        plot_pc_relationships(df_data, OUTPUT_DIR)

        # 4. Prompt user to choose an optimal k based on the silhouette plots
        optimal_k = 0
        while True:
            try:
                k_input = input(f"\nBased on the silhouette plots, please enter the optimal number of clusters (k) from {cluster_range_to_test}: ")
                optimal_k = int(k_input)
                if optimal_k in cluster_range_to_test:
                    break
                else:
                    logger.warning(f"Please enter a value that was tested: {cluster_range_to_test}")
            except ValueError:
                logger.warning("Invalid input. Please enter an integer.")

        # 5. Perform final clustering and analysis with the chosen k.
        analyze_and_plot_clusters(df_data, n_clusters=optimal_k, output_dir=OUTPUT_DIR)

    except FileNotFoundError:
        logger.error(f"File not found: {input_file}. Please run bgg_pca_analysis.py first.")

if __name__ == "__main__":
    main()