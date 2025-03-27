import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the traffic data
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert timestamp and extract time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['month'] = df['timestamp'].dt.month_name()
    
    # Create custom hour categories
    def categorize_hour(hour):
        if 5 <= hour < 10:
            return 'Morning Rush (5-10)'
        elif 10 <= hour < 15:
            return 'Mid-Day (10-15)'
        elif 15 <= hour < 19:
            return 'Evening Rush (15-19)'
        else:
            return 'Night/Late Night (19-5)'
    
    df['hour_category'] = df['hour'].apply(categorize_hour)
    
    return df

# Create comprehensive visualizations
def create_traffic_visualizations(df):
    """
    Create a comprehensive set of visualizations for traffic data
    
    Args:
        df (pd.DataFrame): Preprocessed traffic dataframe
    """
    # Set up the figure with more subplots
    fig, axs = plt.subplots(3, 2, figsize=(20, 25))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Peak Traffic by Hour with Hour Categories
    hour_traffic = df.groupby('hour_category')['vehicle_count'].sum().sort_values(ascending=False)
    sns.barplot(x=hour_traffic.index, y=hour_traffic.values, ax=axs[0, 0], palette='viridis')
    axs[0, 0].set_title('Traffic Volume by Hour Category', fontsize=12)
    axs[0, 0].set_xlabel('Hour Category', fontsize=10)
    axs[0, 0].set_ylabel('Total Vehicle Count', fontsize=10)
    axs[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Intersection Traffic with Detailed Distribution
    intersection_traffic = df.groupby('intersection_name')['vehicle_count'].sum().sort_values(ascending=False)
    sns.barplot(x=intersection_traffic.index, y=intersection_traffic.values, ax=axs[0, 1], palette='rocket')
    axs[0, 1].set_title('Traffic by Intersection', fontsize=12)
    axs[0, 1].set_xlabel('Intersection Name', fontsize=10)
    axs[0, 1].set_ylabel('Total Vehicle Count', fontsize=10)
    axs[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Day of Week Analysis with Statistical Overlay
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_analysis = df.groupby('day_of_week')['vehicle_count'].agg(['sum', 'mean']).reindex(day_order)
    
    ax3 = axs[1, 0]
    day_analysis['sum'].plot(kind='bar', ax=ax3, color='skyblue', alpha=0.7)
    ax3_twin = ax3.twinx()
    day_analysis['mean'].plot(kind='line', ax=ax3_twin, color='red', marker='o')
    
    ax3.set_title('Traffic Volume and Average by Day of Week', fontsize=12)
    ax3.set_xlabel('Day of Week', fontsize=10)
    ax3.set_ylabel('Total Vehicle Count', fontsize=10, color='skyblue')
    ax3_twin.set_ylabel('Average Vehicle Count', fontsize=10, color='red')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # 4. Advanced Heatmap of Vehicle Counts
    hour_day_traffic = df.groupby(['day_of_week', 'hour'])['vehicle_count'].mean().unstack().reindex(day_order)
    sns.heatmap(hour_day_traffic, ax=axs[1, 1], cmap='YlGnBu', 
                annot=True, fmt='.0f', cbar_kws={'label': 'Average Vehicle Count'})
    axs[1, 1].set_title('Average Vehicle Counts by Hour and Day', fontsize=12)
    
    # 5. Weather Condition Impact
    weather_traffic = df.groupby('weather_condition')['vehicle_count'].agg(['mean', 'sum'])
    weather_traffic.plot(kind='bar', ax=axs[2, 0], secondary_y='sum')
    axs[2, 0].set_title('Traffic by Weather Condition', fontsize=12)
    axs[2, 0].set_xlabel('Weather Condition', fontsize=10)
    axs[2, 0].tick_params(axis='x', rotation=45)
    
    # 6. Speed Distribution
    sns.boxplot(x='day_of_week', y='average_speed', data=df, ax=axs[2, 1], palette='Set3', order=day_order)
    axs[2, 1].set_title('Speed Distribution by Day of Week', fontsize=12)
    axs[2, 1].set_xlabel('Day of Week', fontsize=10)
    axs[2, 1].set_ylabel('Average Speed', fontsize=10)
    axs[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Comprehensive Statistical Analysis
def perform_statistical_analysis(df):
    """
    Perform comprehensive statistical analysis on the dataset
    
    Args:
        df (pd.DataFrame): Preprocessed traffic dataframe
    """
    # Numerical Columns Analysis
    numerical_columns = ['vehicle_count', 'average_speed']
    stats_summary = {}
    for column in numerical_columns:
        stats_summary[column] = {
            'Mean': df[column].mean(),
            'Median': df[column].median(),
            'Standard Deviation': df[column].std(),
            'Min': df[column].min(),
            'Max': df[column].max(),
            'Skewness': df[column].skew(),
            'Kurtosis': df[column].kurtosis()
        }
    
    print("\n--- Detailed Statistical Summary ---")
    for column, stats in stats_summary.items():
        print(f"\n{column.replace('_', ' ').title()} Statistics:")
        for stat, value in stats.items():
            print(f"{stat}: {value:.2f}")
    
    # Categorical Variables Analysis
    categorical_columns = ['weather_condition', 'day_of_week', 'intersection_name']
    print("\n--- Categorical Variables Summary ---")
    for column in categorical_columns:
        print(f"\n{column.replace('_', ' ').title()} Distribution:")
        summary = df[column].value_counts(normalize=True) * 100
        print(summary.round(2))

# Main execution
def main(file_path):
    """
    Main function to orchestrate data analysis
    
    Args:
        file_path (str): Path to the traffic data CSV
    """
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    
    # Create visualizations
    create_traffic_visualizations(df)
    
    # Perform statistical analysis
    perform_statistical_analysis(df)

# Run the analysis (replace with your actual file path)
main(r"C:\Users\Ayush\Desktop\TraFFIC DATA VISION\Prayagraj_Traffic_Data.csv")
