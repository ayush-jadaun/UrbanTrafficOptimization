import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv(r"C:\Users\Ayush\Desktop\TraFFIC DATA VISION\Prayagraj_Traffic_Data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

fig, axs = plt.subplots(3, 2, figsize=(15, 20))

peak_traffic = df.groupby('hour')['vehicle_count'].sum().reset_index()
sns.barplot(data=peak_traffic, x='hour', y='vehicle_count', ax=axs[0, 0], hue='hour', palette='viridis', legend=False)
axs[0, 0].set_title('Peak Traffic by Hour')
axs[0, 0].set_xlabel('Hour of Day')
axs[0, 0].set_ylabel('Total Vehicle Count')

intersection_traffic = df.groupby('intersection_name')['vehicle_count'].sum().reset_index()
sns.barplot(data=intersection_traffic, x='intersection_name', y='vehicle_count', ax=axs[0, 1], hue='intersection_name', palette='viridis', legend=False)
axs[0, 1].set_title('Traffic by Intersection')
axs[0, 1].set_xlabel('Intersection Name')
axs[0, 1].set_ylabel('Total Vehicle Count')
axs[0, 1].tick_params(axis='x', rotation=45)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_analysis = df.groupby('day_of_week')['vehicle_count'].sum().reindex(day_order).reset_index()
sns.barplot(data=day_analysis, x='day_of_week', y='vehicle_count', ax=axs[1, 0], hue='day_of_week', palette='viridis', legend=False)
axs[1, 0].set_title('Traffic Analysis by Day of the Week')
axs[1, 0].set_xlabel('Day of Week')
axs[1, 0].set_ylabel('Total Vehicle Count')

hour_day_traffic = df.groupby(['day_of_week', 'hour'])['vehicle_count'].sum().unstack().reindex(day_order)
sns.heatmap(hour_day_traffic, ax=axs[1, 1], cmap='viridis', cbar_kws={'label': 'Total Vehicle Count'})
axs[1, 1].set_title('Heatmap of Vehicle Counts by Hour and Day of Week')
axs[1, 1].set_xlabel('Hour of Day')
axs[1, 1].set_ylabel('Day of Week')

axs[2, 0].pie(intersection_traffic['vehicle_count'], labels=intersection_traffic['intersection_name'], autopct='%1.1f%%', startangle=90)
axs[2, 0].set_title('Vehicle Distribution by Intersection')

axs[2, 1].axis('off')
plt.tight_layout()
plt.show()

numerical_columns = ['vehicle_count', 'average_speed']
stats_summary = {}

for column in numerical_columns:
    stats_summary[column] = {
        'Mean': df[column].mean(),
        'Median': df[column].median(),
        'Mode': df[column].mode()[0],
        'Min': df[column].min(),
        'Max': df[column].max()
    }

print("Basic Statistics for Numerical Columns:")
for column, stats in stats_summary.items():
    print(f"\n{column}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")

categorical_columns = ['weather_condition', 'day_of_week']
categorical_summary = {}

for column in categorical_columns:
    categorical_summary[column] = df[column].value_counts()

print("\nCategorical Variables Summary:")
for column, summary in categorical_summary.items():
    print(f"\n{column}:")
    print(summary)
