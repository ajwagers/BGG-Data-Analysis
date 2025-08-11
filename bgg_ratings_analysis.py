import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import logging
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("ratings_analysis_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

def analyze_rating_distribution(filepath: str, output_dir: Path):
    """
    Reproduces the rating distribution analysis from the "Loaded Dice" article by
    plotting the distribution of user ratings against a normal curve.
    """
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}. Please run bgg_data_ingestion.py again to generate the required vote count columns.")
        return

    vote_cols = [f'votes_{i}' for i in range(1, 11)]
    if not all(col in df.columns for col in vote_cols):
        logger.error(f"One or more required columns ({', '.join(vote_cols)}) not found in {filepath}.")
        logger.error("Please re-run bgg_data_ingestion.py to generate the data with rating distributions.")
        return

    # --- 1. Aggregate vote counts across all games ---
    logger.info("Aggregating rating distribution across all games...")
    total_votes_per_level = df[vote_cols].sum()
    total_votes_per_level.index = range(1, 11) # Set index to be the rating level (1-10)
    
    # --- 2. Calculate statistics of the distribution ---
    total_num_votes = total_votes_per_level.sum()
    if total_num_votes == 0:
        logger.error("No rating data found in the file. Aborting analysis.")
        return
        
    # Calculate mean
    weighted_sum = (total_votes_per_level.index * total_votes_per_level).sum()
    mean_rating = weighted_sum / total_num_votes
    
    # Calculate standard deviation
    variance = (( (total_votes_per_level.index - mean_rating)**2 * total_votes_per_level ).sum()) / total_num_votes
    std_dev = np.sqrt(variance)
    
    logger.info(f"Distribution Statistics: Mean={mean_rating:.4f}, Std Dev={std_dev:.4f}, Total Votes={int(total_num_votes)}")

    # --- 3. Plot the distribution ---
    logger.info("Generating rating distribution plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bar plot for the actual distribution
    total_votes_per_level.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black', label='Actual User Ratings')
    ax.set_xlabel("User Rating")
    ax.set_ylabel("Number of Votes")
    ax.set_title("Distribution of BGG User Ratings vs. Normal Distribution", fontsize=16)
    ax.tick_params(axis='x', rotation=0)
    
    # --- 4. Overlay the normal distribution ---
    # Generate points for the normal curve
    x_norm = np.linspace(0.5, 10.5, 1000)
    # The normal distribution gives a probability density. To overlay it on a count plot,
    # we need to scale it by the total number of votes and the width of the histogram bins (which is 1).
    y_norm = norm.pdf(x_norm, mean_rating, std_dev) * total_num_votes

    ax.plot(x_norm - 1, y_norm, 'r-', linewidth=2, label=f'Normal Distribution (μ={mean_rating:.2f}, σ={std_dev:.2f})')
    
    # --- 5. Final styling ---
    ax.legend()
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plot_filename = output_dir / "bgg_rating_distribution.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    logger.info(f"Plot saved to {plot_filename}")
    plt.close(fig)

def main():
    """Main function to run the ratings analysis."""
    input_file = 'bgg_top_games_updated.csv'
    analyze_rating_distribution(input_file, OUTPUT_DIR)

if __name__ == "__main__":
    main()
