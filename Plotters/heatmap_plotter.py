import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

def plot_spectrum(df, x_point, y_point, plot_type):
    print(f"Looking for data at (x={x_point}, y={y_point}) in the dataframe.")
    point_data = df[(df['x'] == y_point) & (df['y'] == x_point)]
    if not point_data.empty:
        plt.figure(figsize=(8, 6))
        plt.plot(point_data['wavenumber_energy'], point_data['intensity'], marker='o', markersize=1, linestyle='-', color='r')
        plt.xlabel('Raman shift (cm-1)' if plot_type == 'Raman' else 'Energy (eV)')
        plt.ylabel('Intensity')
        plt.title(f'Intensity at (x={x_point}, y={y_point})')
        plt.show()
    else:
        print(f"No data found at (x={x_point}, y={y_point})")

def on_click(event, heatmap_data, df, plot_type):
    if event.xdata is None or event.ydata is None:
        print("Click outside axes bounds. Ignoring click.")
        return
    
    col = int(event.xdata + 0.5)
    row = int(event.ydata + 0.5)

    print(f"Clicked at col: {col}, row: {row}")
    
    if 0 <= col < heatmap_data.shape[1] and 0 <= row < heatmap_data.shape[0]:
        x_point = heatmap_data.columns[col]
        y_point = heatmap_data.index[row]
        print(f"Plotting spectrum at (x={x_point}, y={y_point})")
        plot_spectrum(df, x_point, y_point, plot_type)
    else:
        print(f"Click out of data bounds (col: {col}, row: {row}). Ignoring click.")

def plot_heatmap(data_folder, txt_files, wavenumber_min=1.6, wavenumber_max=2.2, plot_type='PL', output_folder=None,usernormalized=True):
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
        unormalized_data = heatmap_data.copy()
        heatmap_data = (heatmap_data - heatmap_data.min().min()) / (heatmap_data.max().max() - heatmap_data.min().min())

        plt.figure(figsize=(10, 8))
        if usernormalized:
                ax = sns.heatmap(heatmap_data, cmap='plasma', cbar=True, cbar_kws={'label': 'PL Intensity' if plot_type == 'PL' else 'Raman Intensity'}, vmin=0, vmax=1)
        else:
                ax = sns.heatmap(unormalized_data, cmap='plasma', cbar=True, cbar_kws={'label': 'PL Intensity' if plot_type == 'PL' else 'Raman Intensity'})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10) 
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xticks([])
        plt.yticks([])
        
        plt.title(f'Heat Map of {"PL" if plot_type == "PL" else "Raman"} Intensity')
        output_path = os.path.join(output_folder, f'{os.path.splitext(txt_file)[0]}.png')
        plt.savefig(output_path, dpi=300)
        # 添加鼠标悬停显示坐标功能
        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < heatmap_data.shape[1] and row >= 0 and row < heatmap_data.shape[0]:
                abs_x = heatmap_data.columns[col]
                abs_y = heatmap_data.index[row]
                z = heatmap_data.iloc[row, col]
                return f'x={abs_x}, y={abs_y}, intensity={z:.2f}'
            else:
                return f'x={col}, y={row}'

        ax.format_coord = format_coord
        # Bind the click event
        ax.figure.canvas.mpl_connect('button_press_event', lambda event: on_click(event, heatmap_data, df, plot_type))

        plt.show()
        plt.close()

def browse_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
    if file_paths:
        entry_file_paths.delete(0, tk.END)
        entry_file_paths.insert(0, ";".join(file_paths))

def browse_output_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        entry_output_folder.delete(0, tk.END)
        entry_output_folder.insert(0, folder_path)

def run_plot():
    file_paths = entry_file_paths.get().split(";")
    data_folder = os.path.dirname(file_paths[0])
    txt_files = [os.path.basename(file_path) for file_path in file_paths]
    wavenumber_min = float(entry_wavenumber_min.get())
    wavenumber_max = float(entry_wavenumber_max.get())
    plot_type = var_plot_type.get()
    output_folder = entry_output_folder.get()
    use_normalized = var_normalize.get()

    try:
        plot_heatmap(data_folder, txt_files, wavenumber_min, wavenumber_max, plot_type, output_folder,use_normalized)
        messagebox.showinfo("Success", "Plots created successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# 创建主窗口
root = tk.Tk()
root.title("Heatmap Plotter")

# 文件路径输入框
tk.Label(root, text="File Paths:").grid(row=0, column=0, padx=10, pady=5)
entry_file_paths = tk.Entry(root, width=50)
entry_file_paths.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=browse_files).grid(row=0, column=2, padx=10, pady=5)

# 输出文件夹路径输入框
tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=10, pady=5)
entry_output_folder = tk.Entry(root, width=50)
entry_output_folder.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=browse_output_folder).grid(row=1, column=2, padx=10, pady=5)

# wavenumber_min 输入框
tk.Label(root, text="Wavenumber Min:").grid(row=2, column=0, padx=10, pady=5)
entry_wavenumber_min = tk.Entry(root)
entry_wavenumber_min.grid(row=2, column=1, padx=10, pady=5)
entry_wavenumber_min.insert(0, "1.6")

# wavenumber_max 输入框
tk.Label(root, text="Wavenumber Max:").grid(row=3, column=0, padx=10, pady=5)
entry_wavenumber_max = tk.Entry(root)
entry_wavenumber_max.grid(row=3, column=1, padx=10, pady=5)
entry_wavenumber_max.insert(0, "2.2")

# plot_type 选择框
tk.Label(root, text="Plot Type:").grid(row=4, column=0, padx=10, pady=5)
var_plot_type = tk.StringVar(value="PL")
tk.OptionMenu(root, var_plot_type, "PL", "Raman").grid(row=4, column=1, padx=10, pady=5)

# 添加选择框用于选择是否使用归一化强度
tk.Label(root, text="Use Normalized Intensity:").grid(row=5, column=0, padx=10, pady=5)
var_normalize = tk.BooleanVar(value=True)
tk.Checkbutton(root, variable=var_normalize).grid(row=5, column=1, padx=10, pady=5)

# 运行按钮
tk.Button(root, text="Run", command=run_plot).grid(row=6, column=0, columnspan=3, pady=10)

# 运行主循环
root.mainloop()