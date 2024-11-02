import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\Ayush\Desktop\TraFFIC DATA VISION\Prayagraj_Traffic_Data.csv")

# Print the first few rows of the dataset to ensure it's loaded correctly
print("Initial DataFrame:")
print(df.head())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract hour and day of the week from the timestamp
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# Print the updated DataFrame to check the new columns
print("\nDataFrame after adding hour and day of week:")
print(df.head())

# Prepare the figure and axes for subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 20))  # Adjusted height

# 1. Peak Traffic by Hour
peak_traffic = df.groupby('hour')['vehicle_count'].sum().reset_index()
print("\nPeak Traffic Data:")
print(peak_traffic)
sns.barplot(data=peak_traffic, x='hour', y='vehicle_count', ax=axs[0, 0], hue='hour', palette='viridis', legend=False)
axs[0, 0].set_title('Peak Traffic by Hour')
axs[0, 0].set_xlabel('Hour of Day')
axs[0, 0].set_ylabel('Total Vehicle Count')

# 2. Traffic by Intersection
intersection_traffic = df.groupby('intersection_name')['vehicle_count'].sum().reset_index()
print("\nTraffic by Intersection Data:")
print(intersection_traffic)
sns.barplot(data=intersection_traffic, x='intersection_name', y='vehicle_count', ax=axs[0, 1], hue='intersection_name', palette='viridis', legend=False)
axs[0, 1].set_title('Traffic by Intersection')
axs[0, 1].set_xlabel('Intersection Name')
axs[0, 1].set_ylabel('Total Vehicle Count')
axs[0, 1].tick_params(axis='x', rotation=45)

# 3. Traffic Analysis by Day of the Week
day_analysis = df.groupby('day_of_week')['vehicle_count'].sum().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
print("\nTraffic Analysis by Day of the Week Data:")
print(day_analysis)
sns.barplot(data=day_analysis, x='day_of_week', y='vehicle_count', ax=axs[1, 0], hue='day_of_week', palette='viridis', legend=False)
axs[1, 0].set_title('Traffic Analysis by Day of the Week')
axs[1, 0].set_xlabel('Day of Week')
axs[1, 0].set_ylabel('Total Vehicle Count')

# 4. Heatmap for Hour vs. Day of Week Vehicle Counts
hour_day_traffic = df.groupby(['day_of_week', 'hour'])['vehicle_count'].sum().unstack()
hour_day_traffic = hour_day_traffic.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

print("\nHeatmap Data (Hour vs. Day of Week):")
print(hour_day_traffic)
sns.heatmap(hour_day_traffic, ax=axs[1, 1], cmap='viridis', annot=False, cbar_kws={'label': 'Total Vehicle Count'})
axs[1, 1].set_title('Heatmap of Vehicle Counts by Hour and Day of Week')
axs[1, 1].set_xlabel('Hour of Day')
axs[1, 1].set_ylabel('Day of Week')

# 5. Pie Chart for Vehicle Distribution
axs[2, 0].pie(intersection_traffic['vehicle_count'], labels=intersection_traffic['intersection_name'], autopct='%1.1f%%', startangle=90)
axs[2, 0].set_title('Vehicle Distribution by Intersection')

# Hide the last subplot if unused
axs[2, 1].axis('off')  # Hide unused subplot

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
