import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load your data
df = pd.read_csv('map_raman_RT.txt', sep='\s+', header=None)  # header=None treats the first row as data

# Filter the DataFrame to include wavelengths between 500 and 540
filtered_df = df[(df[2] >= 500) & (df[2] <= 540)]

# Group the DataFrame by X and Y coordinates and mean the Intensity values
grouped_df = filtered_df.groupby([0, 1])[3].mean().reset_index()

# Create a pivot table with X, Y as index and the Intensity as the column
pivot_df = pd.pivot_table(grouped_df, values=3, index=[0, 1], aggfunc='mean')

# Create a heat map for the Intensity
plt.figure(figsize=(14, 6))
sns.heatmap(pivot_df.unstack(), cmap='plasma', annot=False, fmt='.2f', cbar_kws={'label': 'Intensity'})
plt.title('Heat Map of Intensity (500-540cm-1)')
plt.xlabel('Y')
plt.ylabel('X')
plt.show()