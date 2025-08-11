import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from itertools import combinations
from collections import Counter

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("mechanics_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Loads and preprocesses the data for mechanics analysis."""
    logger.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}. Please run bgg_data_ingestion.py first.")
        return None

    # --- Data Cleaning ---
    # Drop rows where essential data is missing
    df.dropna(subset=['year_published', 'mechanics', 'average_weight'], inplace=True)
    
    # Convert year to integer
    df['year_published'] = df['year_published'].astype(int)
    
    # Split the mechanics string into a list of individual mechanics
    df['mechanics_list'] = df['mechanics'].str.split(';').apply(lambda x: [m.strip() for m in x if m.strip()])
    
    # Add a column for the number of mechanics
    df['num_mechanics'] = df['mechanics_list'].apply(len)
    
    # Filter out games with no mechanics listed
    df = df[df['num_mechanics'] > 0].copy()
    
    logger.info(f"Data loaded and prepared. Shape: {df.shape}")
    return df

def plot_avg_mechanics_over_time(df: pd.DataFrame, output_dir: Path):
    """
    Calculates and plots the average number of mechanics per game over time,
    along with the number of games published each year.
    """
    logger.info("Analyzing average number of mechanics over time...")
    
    # Group by year and calculate the mean number of mechanics and count of games
    yearly_stats = df.groupby('year_published').agg(
        avg_num_mechanics=('num_mechanics', 'mean'),
        num_games=('id', 'count')
    ).reset_index()
    
    # Filter for a reasonable time range to avoid noise from very old years with few games
    yearly_stats = yearly_stats[yearly_stats['year_published'] >= 1980]
    
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # Plot 1: Average number of mechanics (line plot on primary y-axis)
    sns.lineplot(data=yearly_stats, x='year_published', y='avg_num_mechanics', ax=ax1, color='dodgerblue', label='Avg. Mechanics per Game')
    sns.regplot(data=yearly_stats, x='year_published', y='avg_num_mechanics', scatter=False, ax=ax1, color='red', line_kws={'linestyle':'--'}, label='Trendline')
    ax1.set_xlabel('Year Published', fontsize=12)
    ax1.set_ylabel('Average Number of Mechanics', color='dodgerblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.grid(True, which='major', axis='y',
    linestyle='--', linewidth=0.5)

    # Plot 2: Number of games (bar plot on secondary y-axis)
    ax2 = ax1.twinx()
    sns.barplot(data=yearly_stats, x='year_published', y='num_games', ax=ax2, color='lightgrey', alpha=0.6, label='Games Published')
    ax2.set_ylabel('Number of Games Published', color='grey', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.set_ylim(0, yearly_stats['num_games'].max() * 1.1) # Give some headroom

    # Final plot adjustments
    plt.title('Average Mechanics per Game & Number of Games Published Over Time', fontsize=16)
    fig.tight_layout()
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax2.legend(lines + bars, labels + bar_labels, loc='upper left')
    
    filename = output_dir / "avg_mechanics_over_time.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()
def plot_weight_vs_mechanics(df: pd.DataFrame, output_dir: Path):
    """Analyzes and plots the relationship between game weight and number of mechanics."""
    logger.info("Analyzing game weight vs. number of mechanics...")
    
    # Bin the game weights into discrete groups
    # Bins: 0-0.5, 0.5-1.0, ..., 4.5-5
    bins = np.arange(0, 5.1, 0.5)
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    df['weight_group'] = pd.cut(df['average_weight'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    # Group by weight and calculate the average number of mechanics
    weight_mechanics = df.groupby('weight_group')['num_mechanics'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=weight_mechanics, x='weight_group', y='num_mechanics', palette='viridis')
    
    plt.title('Average Number of Mechanics vs. Game Weight (Complexity)', fontsize=16)
    plt.xlabel('Game Weight Group')
    plt.ylabel('Average Number of Mechanics')
    plt.xticks(rotation=45)
    
    filename = output_dir / "weight_vs_mechanics.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()

def plot_mechanic_frequency(df: pd.DataFrame, output_dir: Path) -> pd.Series:
    """Counts and plots the frequency of the top 20 mechanics."""
    logger.info("Analyzing mechanic frequency...")
    
    # Explode the list of mechanics into one row per mechanic per game
    all_mechanics = df.explode('mechanics_list')['mechanics_list']
    
    # Count the occurrences of each mechanic
    mechanic_counts = all_mechanics.value_counts()
    
    # Plot the top 20
    top_20_mechanics = mechanic_counts.head(20)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=top_20_mechanics.values, y=top_20_mechanics.index, palette='mako')
    
    plt.title('Top 20 Most Frequent Game Mechanics', fontsize=16)
    plt.xlabel('Number of Games')
    plt.ylabel('Mechanic')
    
    filename = output_dir / "top_20_mechanics.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()
    
    return mechanic_counts

def plot_top_mechanics_trends(df: pd.DataFrame, mechanic_counts: pd.Series, output_dir: Path):
    """Plots the popularity trend of the top 5 mechanics and Card Drafting over time."""
    logger.info("Analyzing trends of top mechanics...")

    # Get top 5 and add 'Card Drafting' for comparison
    top_5_mechanics = mechanic_counts.head(5).index.tolist()
    mechanics_to_plot = top_5_mechanics + ['Card Drafting']
    # Remove duplicates if 'Card Drafting' is already in the top 5, preserving order
    mechanics_to_plot = list(dict.fromkeys(mechanics_to_plot))

    # Filter for a reasonable time range
    df_filtered = df[df['year_published'] >= 1980].copy()

    yearly_game_counts = df_filtered['year_published'].value_counts().sort_index()

    trends_list = []
    for mechanic in mechanics_to_plot:
        # Check if a game contains the mechanic
        has_mechanic = df_filtered['mechanics_list'].apply(lambda x: mechanic in x)

        # Group by year and sum the boolean column to get the count
        mechanic_yearly_count = df_filtered[has_mechanic].groupby('year_published').size()

        # Calculate percentage and create a DataFrame for this mechanic
        mechanic_yearly_percent = (mechanic_yearly_count / yearly_game_counts * 100).rename(mechanic)
        trends_list.append(mechanic_yearly_percent)

    trends_df = pd.concat(trends_list, axis=1).fillna(0)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    trends_df.plot(ax=ax)

    plt.title('Popularity Trend of Key Mechanics Over Time', fontsize=16)
    plt.xlabel('Year Published')
    plt.ylabel('Percentage of Games Published That Year (%)')
    plt.legend(title='Mechanic')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    filename = output_dir / "top_5_mechanics_trends.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close('all') # Close all figures, as the plot() call on df can create one.

def analyze_mechanic_pairs(df: pd.DataFrame, mechanic_counts: pd.Series, output_dir: Path):
    """Analyzes co-occurrence of mechanic pairs."""
    logger.info("Analyzing two-mechanic combinations...")
    
    total_games = len(df)
    
    # --- Calculate Observed Frequencies ---
    # Generate all pairs for each game and count them
    pair_counter = Counter()
    for mechanics_list in df['mechanics_list']:
        # Sort to treat (A, B) and (B, A) as the same pair
        sorted_mechanics = sorted(mechanics_list)
        for pair in combinations(sorted_mechanics, 2):
            pair_counter[pair] += 1
    
    observed_freq = pd.Series(pair_counter)
    
    # Filter for pairs that appear a minimum number of times to be significant
    observed_freq = observed_freq[observed_freq >= 10]
    
    # --- Calculate Expected Frequencies ---
    # P(A) = count(A) / total_games
    mechanic_prob = mechanic_counts / total_games
    
    expected_freq = {}
    for pair in observed_freq.index:
        mech1, mech2 = pair
        # Expected count = P(A) * P(B) * N
        expected_count = mechanic_prob[mech1] * mechanic_prob[mech2] * total_games
        expected_freq[pair] = expected_count
        
    expected_freq = pd.Series(expected_freq)
    
    # --- Compare Observed vs. Expected ---
    lift_df = pd.DataFrame({'Observed': observed_freq, 'Expected': expected_freq})
    lift_df['Lift'] = lift_df['Observed'] / lift_df['Expected']
    
    # Sort to find most over- and under-represented pairs
    overrepresented = lift_df.sort_values(by='Lift', ascending=False).head(20)
    underrepresented = lift_df.sort_values(by='Lift', ascending=True).head(20)
    
    # --- Plotting ---
    def plot_lift(data: pd.DataFrame, title: str, filename: str):
        data.index = [f"{p[0]} & {p[1]}" for p in data.index]
        data = data.sort_values(by='Lift', ascending=True) # For horizontal bar plot

        plt.figure(figsize=(12, 10))
        # Use a diverging palette to show over/under representation clearly
        sns.barplot(x=data['Lift'], y=data.index, palette='coolwarm_r')
        
        # Add a line at Lift=1.0, which is the boundary for no correlation
        plt.axvline(x=1.0, color='black', linestyle='--')
        
        plt.title(title, fontsize=16)
        plt.xlabel('Lift (Observed / Expected Frequency)')
        plt.ylabel('Mechanic Pair')
        
        filepath = output_dir / filename
        plt.savefig(filepath, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")
        plt.close()

    plot_lift(overrepresented, "Top 20 Overrepresented Mechanic Pairs (Co-occur More Than Expected)", "overrepresented_pairs.png")
    plot_lift(underrepresented, "Top 20 Underrepresented Mechanic Pairs (Co-occur Less Than Expected)", "underrepresented_pairs.png")

def main():
    """Main function to run all mechanics analyses."""
    input_file = 'bgg_top_games_updated.csv'
    
    df = load_and_prepare_data(input_file)
    
    if df is not None:
        logger.info("\n--- Starting Mechanics Analysis ---")
        
        plot_avg_mechanics_over_time(df, OUTPUT_DIR)
        plot_weight_vs_mechanics(df, OUTPUT_DIR)
        mechanic_counts = plot_mechanic_frequency(df, OUTPUT_DIR)
        plot_top_mechanics_trends(df, mechanic_counts, OUTPUT_DIR)
        analyze_mechanic_pairs(df, mechanic_counts, OUTPUT_DIR)
        
        logger.info("\n--- Mechanics Analysis Complete ---")
        logger.info(f"All plots saved in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()