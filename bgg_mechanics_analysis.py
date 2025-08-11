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
    # Drop rows where essential data is missing for any of our analyses
    df.dropna(subset=['year_published', 'mechanics', 'average_weight', 'bgg_rank', 'categories'], inplace=True)
    
    # Convert year to integer
    df['year_published'] = df['year_published'].astype(int)
    
    # Split the mechanics string into a list of individual mechanics
    df['mechanics_list'] = df['mechanics'].str.split(';').apply(lambda x: [m.strip() for m in x if m.strip()])
    
    # Add a column for the number of mechanics
    df['num_mechanics'] = df['mechanics_list'].apply(len)

    # Extract primary category to use as genre
    df['main_category'] = df['categories'].str.split(';').str[0].str.strip()
    
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
    yearly_stats = yearly_stats[yearly_stats['year_published'] >= 1980].copy()

    # Calculate a 5-year moving average to smooth the trendline
    yearly_stats['moving_avg'] = yearly_stats['avg_num_mechanics'].rolling(window=5, center=True, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(15, 7))    

    # Plot the background bars first on a secondary y-axis
    ax2 = ax1.twinx()
    # Use plt.bar instead of sns.barplot to ensure the x-axis is treated numerically, aligning it with the line plot.
    ax2.bar(yearly_stats['year_published'], yearly_stats['num_games'], color='lightgrey', alpha=0.6, label='Games Published')
    ax2.set_ylabel('Number of Games Published', color='grey', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='grey')
    # Set the y-axis limit for the bar graph to a fixed value
    ax2.set_ylim(0, 2200)
    ax2.grid(False) # Turn off grid for the background axis

    # Plot the foreground lines on the primary y-axis so they appear on top
    sns.lineplot(data=yearly_stats, x='year_published', y='avg_num_mechanics', ax=ax1, color='dodgerblue', label='Avg. Mechanics per Game')
    sns.lineplot(data=yearly_stats, x='year_published', y='moving_avg', ax=ax1, color='red', linestyle='--', label='5-Year Moving Average')
    ax1.set_xlabel('Year Published', fontsize=12)
    ax1.set_ylabel('Average Number of Mechanics', color='dodgerblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

    # Final plot adjustments
    plt.title('Average Mechanics per Game & Number of Games Published Over Time', fontsize=16)
    fig.tight_layout()
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc='upper left')
    
    filename = output_dir / "avg_mechanics_over_time.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()

def plot_games_published_per_year(df: pd.DataFrame, output_dir: Path):
    """Calculates and plots the number of games published per year."""
    logger.info("Analyzing number of games published per year...")
    
    # Group by year and count the number of games
    yearly_counts = df.groupby('year_published').size().reset_index(name='num_games')
    
    # Filter for a reasonable time range
    yearly_counts = yearly_counts[yearly_counts['year_published'] >= 1980]
    
    plt.figure(figsize=(15, 7))
    ax = sns.barplot(data=yearly_counts, x='year_published', y='num_games', color='steelblue')
    
    plt.title('Number of Games Published Per Year', fontsize=16)
    plt.xlabel('Year Published')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=90)
    
    # Make x-axis labels more readable by showing every 5th year
    for index, label in enumerate(ax.get_xticklabels()):
        if index % 5 != 0:
            label.set_visible(False)

    filename = output_dir / "games_published_per_year.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()

def plot_num_mechanics_distribution(df: pd.DataFrame, output_dir: Path):
    """
    Calculates and plots the distribution of the number of mechanics per game.
    """
    logger.info("Analyzing distribution of the number of mechanics...")

    # Calculate the percentage of games for each count of mechanics
    dist_series = df['num_mechanics'].value_counts(normalize=True).mul(100).sort_index()
    dist_df = dist_series.reset_index()
    dist_df.columns = ['num_mechanics', 'percentage']

    # For better visualization, cap the number of mechanics shown if it's very high
    max_mechanics_to_plot = 30
    dist_df = dist_df[dist_df['num_mechanics'] <= max_mechanics_to_plot]

    plt.figure(figsize=(15, 8))
    ax = sns.barplot(data=dist_df, x='num_mechanics', y='percentage', color='teal')

    plt.title('Distribution of the Number of Mechanics per Game', fontsize=16)
    plt.xlabel('Number of Mechanics in a Game')
    plt.ylabel('Percentage of Games (%)')
    
    # Add percentage labels on top of each bar for clarity
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    filename = output_dir / "num_mechanics_distribution.png"
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

def plot_top_10_mechanic_trends_by_year(df: pd.DataFrame, output_dir: Path):
    """Analyzes and plots the frequency of mechanics in top 10 ranked games since 2000."""
    logger.info("Analyzing mechanic frequency in top 10 games per year (since 2000)...")

    # Filter for years 2000 and later
    df_recent = df[df['year_published'] >= 2000].copy()

    # Sort by year and rank to easily get the top 10
    df_sorted = df_recent.sort_values(by=['year_published', 'bgg_rank'], ascending=[True, True])

    # Get the top 10 ranked games for each year
    top_10_per_year = df_sorted.groupby('year_published').head(10)

    # Explode the mechanics list and count the frequency of each mechanic in this elite set
    mechanic_counts_in_top_games = top_10_per_year.explode('mechanics_list')['mechanics_list'].value_counts()

    # Plot the top 25 most frequent mechanics
    top_n = 25
    plt.figure(figsize=(12, 12))
    sns.barplot(x=mechanic_counts_in_top_games.head(top_n).values, y=mechanic_counts_in_top_games.head(top_n).index, palette='rocket')
    plt.title(f'Top {top_n} Most Frequent Mechanics in Top 10 Games/Year (Since 2000)', fontsize=16)
    plt.xlabel('Total Appearances in Top 10s')
    plt.ylabel('Mechanic')
    filename = output_dir / "mechanics_in_top_10_games.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()

def plot_mechanics_by_weight_trend(df: pd.DataFrame, output_dir: Path):
    """
    Plots the trend of the mean number of mechanics over time, grouped by game weight.
    """
    logger.info("Analyzing mechanics by weight trend over time...")

    # Filter data for the relevant time period
    df_filtered = df[df['year_published'] >= 2000].copy()

    # Create integer-based weight groups (0-1, 1-2, etc.)
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
    df_filtered['weight_group_int'] = pd.cut(df_filtered['average_weight'], bins=bins, labels=labels, right=False, include_lowest=True)

    # Group by year and the new weight group, then calculate the mean number of mechanics
    trend_data = df_filtered.groupby(['year_published', 'weight_group_int'])['num_mechanics'].mean().reset_index()
    trend_data.rename(columns={'num_mechanics': 'mean_mechanics'}, inplace=True)

    # Define the custom color mapping as requested
    color_map = {
        '0-1': 'orange',
        '1-2': 'red',
        '2-3': 'pink',
        '3-4': 'purple',
        '4-5': 'navy'
    }

    # Create the plot
    plt.figure(figsize=(12, 7))
    ax = sns.lineplot(data=trend_data, x='year_published', y='mean_mechanics', hue='weight_group_int', palette=color_map, hue_order=labels)

    # Style the plot according to the requirements
    plt.title('Mechanics by Weight Trend', fontsize=16)
    plt.xlabel('Year Published')
    plt.ylabel('Mean Mechanics')
    plt.legend(title='Weight Group', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)

    filename = output_dir / "mechanics_by_weight_trend.png"
    plt.savefig(filename, bbox_inches='tight')
    logger.info(f"Plot saved to {filename}")
    plt.close()

def save_top_games_by_year_to_csv(df: pd.DataFrame):
    """Identifies the top 5 ranked games for each year and saves them to a CSV file."""
    logger.info("Analyzing top 5 ranked games per year...")

    # We only need years with at least 5 games to have a meaningful top 5
    year_counts = df['year_published'].value_counts()
    valid_years = year_counts[year_counts >= 5].index
    df_filtered = df[df['year_published'].isin(valid_years)].copy()

    # Sort by year (descending) and rank (ascending)
    df_sorted = df_filtered.sort_values(by=['year_published', 'bgg_rank'], ascending=[False, True])

    # Get the top 5 games for each year
    top_games_by_year = df_sorted.groupby('year_published').head(5)

    # Select and format the columns for the final table
    result_table = top_games_by_year[[
        'year_published',
        'bgg_rank',
        'primary_name',
        'main_category',
        'mechanics'
    ]].copy()

    # Rename columns for clarity
    result_table.rename(columns={
        'year_published': 'Year',
        'bgg_rank': 'Rank',
        'primary_name': 'Game',
        'main_category': 'Genre',
        'mechanics': 'Mechanics'
    }, inplace=True)

    # Convert rank to integer for cleaner display
    result_table['Rank'] = result_table['Rank'].astype(int)

    # Save the resulting table to a CSV file
    output_csv_path = "top_5_games_by_year.csv"
    try:
        result_table.to_csv(output_csv_path, index=False, encoding='utf-8')
        logger.info(f"Top 5 games by year table saved to {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save top games table to {output_csv_path}: {e}")

def main():
    """Main function to run all mechanics analyses."""
    input_file = 'bgg_top_games_updated.csv'
    
    df = load_and_prepare_data(input_file)
    
    if df is not None:
        logger.info("\n--- Starting Mechanics Analysis ---")
        
        plot_games_published_per_year(df, OUTPUT_DIR)
        plot_avg_mechanics_over_time(df, OUTPUT_DIR)
        plot_num_mechanics_distribution(df, OUTPUT_DIR)
        plot_weight_vs_mechanics(df, OUTPUT_DIR)
        mechanic_counts = plot_mechanic_frequency(df, OUTPUT_DIR)
        plot_top_mechanics_trends(df, mechanic_counts, OUTPUT_DIR)
        analyze_mechanic_pairs(df, mechanic_counts, OUTPUT_DIR)
        save_top_games_by_year_to_csv(df)
        plot_top_10_mechanic_trends_by_year(df, OUTPUT_DIR)
        plot_mechanics_by_weight_trend(df, OUTPUT_DIR)
        
        logger.info("\n--- Mechanics Analysis Complete ---")
        logger.info(f"All plots saved in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()