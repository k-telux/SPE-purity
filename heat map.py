import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(data_folder, txt_file=None, wavenumber_min=1.6, wavenumber_max=2.2, plot_type='PL', point_type='max', x_point=None, y_point=None):
    txt_files = [txt_file] if txt_file else [f for f in os.listdir(data_folder) if f.endswith('.txt')]

    for txt_file in txt_files:
        data_path = os.path.join(data_folder, txt_file)
        data = np.loadtxt(data_path, skiprows=1)

        x = data[:, 0]
        y = data[:, 1]
        wavenumber_energy = data[:, 2]
        intensity = data[:, 3]

        mask = (wavenumber_energy >= wavenumber_min) & (wavenumber_energy <= wavenumber_max)
        filtered_data = data[mask]

        df = pd.DataFrame(filtered_data, columns=['x', 'y', 'wavenumber_energy', 'intensity'])

        heatmap_data = df.groupby(['x', 'y'])['intensity'].sum().unstack()
        heatmap_data = (heatmap_data - heatmap_data.min().min()) / (heatmap_data.max().max() - heatmap_data.min().min())

        if point_type == 'max':
            max_intensity_point = heatmap_data.stack().idxmax()
            x_point, y_point = max_intensity_point
        elif point_type == 'min':
            min_intensity_point = heatmap_data.stack().idxmin()
            x_point, y_point = min_intensity_point
        elif point_type == 'given':
            if x_point is None or y_point is None:
                first_point = df.iloc[0]
                x_point = first_point['x']
                y_point = first_point['y']

        point_data = df[(df['x'] == x_point) & (df['y'] == y_point)]
        if not point_data.empty:
            plt.figure(figsize=(8, 6))
            plt.plot(point_data['wavenumber_energy'], point_data['intensity'], marker='o', markersize=1, linestyle='-', color='r')
            plt.xlabel('Raman shift (cm-1)' if plot_type == 'Raman' else 'Energy (eV)')
            plt.ylabel('Intensity')
            plt.title(f'Intensity at (x={x_point}, y={y_point})')
            point_output_path = os.path.join(data_folder, f'{os.path.splitext(txt_file)[0]}_x{x_point}_y{y_point}.png')
            plt.savefig(point_output_path, dpi=300)
            plt.close()

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(heatmap_data, cmap='plasma', cbar=True, cbar_kws={'label': 'PL Intensity' if plot_type == 'PL' else 'Raman Intensity'}, vmin=0, vmax=1)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10) 
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks([])
        plt.yticks([])
        
        plt.title(f'Heat Map of {"PL" if plot_type == "PL" else "Raman"} Intensity')

        output_path = os.path.join(data_folder, f'{os.path.splitext(txt_file)[0]}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()

# The function plot_heatmap_and_wavenumber() takes in the following arguments:
# data_folder: the folder containing the data files
# txt_file: the specific data file to be processed (if None, all .txt files in the folder will be processed)
# wavenumber_min/max: the minimum/maximun wavenumber or energy value to consider
# plot_type: the type of plot to generate ('PL' or 'Raman')
# point_type: the type of point in heatmap to plot ('max', 'min', 'given')
# x_point/y_point: the specific x/y to plot the intensity (if None, the first data point will be used)(optional)
data_folder = 'C:\\Users\\xzqte\\Desktop\\Re-doped'
plot_heatmap(data_folder, 'map_2_pl_1.8eV.txt', wavenumber_min=1.6, wavenumber_max=2.2, plot_type='PL', point_type='max', x_point=None, y_point=None)