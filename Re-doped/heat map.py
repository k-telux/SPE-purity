import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_folder = 'C:\\Users\\xzqte\\Desktop\\Re-doped'

txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

for txt_file in txt_files:
    data_path = os.path.join(data_folder, txt_file)
    data = np.loadtxt(data_path, skiprows=0)

    x = data[:, 0]
    y = data[:, 1]
    wavenumber_energy = data[:, 2]
    intensity = data[:, 3]

    wavenumber_min = 1.6  
    wavenumber_max = 2.2

    mask = (wavenumber_energy >= wavenumber_min) & (wavenumber_energy <= wavenumber_max)
    filtered_data = data[mask]

    df = pd.DataFrame(filtered_data, columns=['x', 'y', 'wavenumber_energy', 'intensity'])
    heatmap_data = df.groupby(['x', 'y'])['intensity'].sum().unstack()
    heatmap_data = (heatmap_data - heatmap_data.min().min()) / (heatmap_data.max().max() - heatmap_data.min().min())

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(heatmap_data, cmap='plasma', cbar=True, cbar_kws={'label': 'PL Intensity'}, vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10) 
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Heat Map of PL Intensity in 1.85 - 1.95 eV')

    output_path = os.path.join(data_folder, f'{os.path.splitext(txt_file)[0]}.png')
    plt.savefig(output_path)
    plt.show()
    plt.close()