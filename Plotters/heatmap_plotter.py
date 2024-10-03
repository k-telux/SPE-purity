import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.ndimage import gaussian_filter1d
from tkinter import filedialog, messagebox, ttk
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.fft import fft, ifft
import math

def ask_save_data():
    root = tk.Tk()
    root.withdraw()  
    result = messagebox.askyesno("Save data", "Save the spectrum of this point to a text file?")
    root.destroy()
    return result

def save_data_to_txt(x_data, y_data, x_point, y_point):
    filename = f"spectrum_data_x{x_point}_y{y_point}.txt"
    with open(filename, 'w') as file:
        for x, y in zip(x_data, y_data):
            file.write(f"{x}\t{y}\n")
    print(f"spectrum is printed to {filename}")

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * wid**2 / ((x-cen)**2 + wid**2)

def double_lorentzian(x, amp1, cen1, wid1, amp2, cen2, wid2):
    return lorentzian(x, amp1, cen1, wid1) + lorentzian(x, amp2, cen2, wid2)

def double_gaussian(x, amp1, cen1, wid1, amp2, cen2, wid2):
    return (amp1 * np.exp(-(x - cen1)**2 / (2 * wid1**2)) +
            amp2 * np.exp(-(x - cen2)**2 / (2 * wid2**2)))

def voigt(x, amp, cen, sigma, gamma):
    z = ((x-cen) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def smooth_fft(y_data, window=10):
    if isinstance(y_data, pd.Series):
        y_data = y_data.values
    y_fft = fft(y_data)
    y_fft[window:] = 0
    y_smooth = ifft(y_fft).real
    return y_smooth

def smooth_gauss(y_data, sigma=2):
    if isinstance(y_data, pd.Series):
        y_data = y_data.values
    y_smooth = gaussian_filter1d(y_data, sigma=sigma)
    return y_smooth

def smooth_move(data, window_size):
    if isinstance(data, pd.Series):
        data = data.values
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_spectrum(df, x_point, y_point, plot_type, perform_fitting, selected_fit=None):
    print(f"Looking for data at (x={x_point}, y={y_point}) in the dataframe. fit: {selected_fit}")
    point_data = df[(df['x'] == y_point) & (df['y'] == x_point)]
    if not point_data.empty:
        x_data = point_data['wavenumber_energy']
        y_data = point_data['intensity']
        
        window_size = 5  
        sigma = 2  
        y_data_smooth = smooth_gauss(y_data, sigma)    
        y_data_smooth = smooth_move(y_data_smooth, window_size)
        x_data_smooth = x_data[:len(y_data_smooth)]
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_data, y_data, marker='o', markersize=1, linestyle='-', color='r', label='Original Data')
        plt.plot(x_data_smooth, y_data_smooth, marker='o', markersize=1, linestyle='-', color='b', label='Smoothed Data')

        if perform_fitting:
            fits = []
            try:
                if selected_fit == 'Gaussian' or selected_fit is None:
                    popt_gauss, _ = curve_fit(gaussian, x_data_smooth, y_data_smooth, p0=[max(y_data_smooth), np.mean(x_data_smooth), np.std(x_data_smooth)])
                    mse_gauss = np.mean((y_data_smooth - gaussian(x_data_smooth, *popt_gauss))**2)
                    fits.append((mse_gauss, gaussian, popt_gauss, 'Gaussian'))

                if selected_fit == 'Lorentzian' or selected_fit is None:
                    popt_lorentz, _ = curve_fit(lorentzian, x_data_smooth, y_data_smooth, p0=[max(y_data_smooth), np.mean(x_data_smooth), np.std(x_data_smooth)])
                    mse_lorentz = np.mean((y_data_smooth - lorentzian(x_data_smooth, *popt_lorentz))**2)
                    fits.append((mse_lorentz, lorentzian, popt_lorentz, 'Lorentzian'))

                if selected_fit == 'Double Lorentz' or selected_fit is None:
                    popt_double_lorentz, _ = curve_fit(double_lorentzian, x_data_smooth, y_data_smooth, p0=[max(y_data_smooth)/2, np.mean(x_data_smooth), np.std(x_data_smooth), max(y_data_smooth)/2, np.mean(x_data_smooth), np.std(x_data_smooth)])
                    mse_double_lorentz = np.mean((y_data_smooth - double_lorentzian(x_data_smooth, *popt_double_lorentz))**2)
                    fits.append((mse_double_lorentz, double_lorentzian, popt_double_lorentz, 'Double Lorentz'))

                if selected_fit == 'Double Gaussian' or selected_fit is None:
                    popt_double_gauss, _ = curve_fit(double_gaussian, x_data_smooth, y_data_smooth, p0=[max(y_data_smooth)/2, np.mean(x_data_smooth), np.std(x_data_smooth), max(y_data_smooth)/2, np.mean(x_data_smooth), np.std(x_data_smooth)])
                    mse_double_gauss = np.mean((y_data_smooth - double_gaussian(x_data_smooth, *popt_double_gauss))**2)
                    fits.append((mse_double_gauss, double_gaussian, popt_double_gauss, 'Double Gaussian'))

                if selected_fit == 'Voigt' or selected_fit is None:
                    popt_voigt, _ = curve_fit(voigt, x_data_smooth, y_data_smooth, p0=[max(y_data_smooth), np.mean(x_data_smooth), np.std(x_data_smooth), np.std(x_data_smooth)])
                    mse_voigt = np.mean((y_data_smooth - voigt(x_data_smooth, *popt_voigt))**2)
                    fits.append((mse_voigt, voigt, popt_voigt, 'Voigt'))
            except RuntimeError as e:
                print(f"Fit error: {e}")

            if fits:
                if selected_fit:
                    fit = next((fit for fit in fits if fit[3] == selected_fit), None)
                    if fit:
                        mse, fit_func, popt, fit_name = fit
                        label = f'{fit_name} Fit: ' + ', '.join([f'{param:.2f}' for param in popt])
                        plt.plot(x_data_smooth, fit_func(x_data_smooth, *popt), label=label, color='g')
                else:
                    best_fit = min(fits, key=lambda x: x[0])
                    mse, fit_func, popt, fit_name = best_fit
                    label = f'{fit_name} Fit: ' + ', '.join([f'{param:.2f}' for param in popt])
                    plt.plot(x_data_smooth, fit_func(x_data_smooth, *popt), label=label, color='g')

        plt.xlabel('Raman shift (cm-1)' if plot_type == 'Raman' else 'Energy (eV)')
        plt.ylabel('Intensity')
        plt.title(f'Intensity at (x={x_point}, y={y_point})')
        plt.legend()
        plt.show()
        
        if ask_save_data():
            save_data_to_txt(x_data, y_data, x_point, y_point)
    else:
        print(f"No data found at (x={x_point}, y={y_point})")

def on_click(event, heatmap_data, df, plot_type, perform_fitting, selected_fit):
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
        plot_spectrum(df, x_point, y_point, plot_type, perform_fitting, selected_fit)
    else:
        print(f"Click out of data bounds (col: {col}, row: {row}). Ignoring click.")

def plot_heatmap(data_folder, txt_files, wavenumber_min=1.6, wavenumber_max=2.2, plot_type='PL', output_folder=None, usernormalized=True, perform_fitting=True, selected_fit=None,log_scale=False):
    for txt_file in txt_files:
        data_path = os.path.join(data_folder, txt_file)
        data = np.loadtxt(data_path, skiprows=1)
        print(f"{selected_fit}")
        x = data[:, 0]
        y = data[:, 1]
        wavenumber_energy = data[:, 2]
        intensity = data[:, 3]

        mask = (wavenumber_energy >= wavenumber_min) & (wavenumber_energy <= wavenumber_max)
        filtered_data = data[mask]

        df = pd.DataFrame(filtered_data, columns=['x', 'y', 'wavenumber_energy', 'intensity'])

        heatmap_data = df.groupby(['x', 'y'])['intensity'].sum().unstack()
        unormalized_data = heatmap_data.copy()
        if log_scale:
            heatmap_data = np.log1p(heatmap_data)
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
        ax.figure.canvas.mpl_connect('button_press_event', lambda event: on_click(event, heatmap_data, df, plot_type, perform_fitting, selected_fit))

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
    selected_fit = var_fit_type.get()
    log_scale = var_log.get()

    try:
        plot_heatmap(data_folder, txt_files, wavenumber_min, wavenumber_max, plot_type, output_folder, use_normalized, selected_fit, log_scale=log_scale)
        messagebox.showinfo("Success", "Plots created successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Heatmap Plotter")

tk.Label(root, text="File Paths:").grid(row=0, column=0, padx=10, pady=5)
entry_file_paths = tk.Entry(root, width=50)
entry_file_paths.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=browse_files).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=10, pady=5)
entry_output_folder = tk.Entry(root, width=50)
entry_output_folder.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=browse_output_folder).grid(row=1, column=2, padx=10, pady=5)

tk.Label(root, text="Wavenumber Min:").grid(row=2, column=0, padx=10, pady=5)
entry_wavenumber_min = tk.Entry(root)
entry_wavenumber_min.grid(row=2, column=1, padx=10, pady=5)
entry_wavenumber_min.insert(0, "1.6")

tk.Label(root, text="Wavenumber Max:").grid(row=3, column=0, padx=10, pady=5)
entry_wavenumber_max = tk.Entry(root)
entry_wavenumber_max.grid(row=3, column=1, padx=10, pady=5)
entry_wavenumber_max.insert(0, "2.2")

tk.Label(root, text="Plot Type:").grid(row=4, column=0, padx=10, pady=5)
var_plot_type = tk.StringVar(value="PL")
ttk.Combobox(root, width=10,textvariable=var_plot_type, values=["PL", "Raman"]).grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Use Normalized Intensity:").grid(row=5, column=0, padx=10, pady=5)
var_normalize = tk.BooleanVar(value=True)
tk.Checkbutton(root, variable=var_normalize).grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Perform Fitting:").grid(row=6, column=0, padx=10, pady=5)
var_fitting = tk.BooleanVar(value=True)
tk.Checkbutton(root, variable=var_fitting).grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Use log scale:").grid(row=7, column=0, padx=10, pady=5)
var_log = tk.BooleanVar(value=False)
tk.Checkbutton(root, variable=var_log).grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Fit Type:").grid(row=8, column=0, padx=10, pady=5)
var_fit_type = tk.StringVar(value="None")
ttk.Combobox(root, width=15, textvariable=var_fit_type, values=["Gaussian", "Lorentzian", "Double Lorentz","Double Gaussian", "Voigt"]).grid(row=8, column=1, padx=10, pady=5)

tk.Button(root, text="Run", command=run_plot).grid(row=9, column=0, columnspan=3, pady=10)

root.mainloop()