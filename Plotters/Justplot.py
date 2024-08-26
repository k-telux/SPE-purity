import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.special import wofz

# 缓冲区保存曲线参数
curve_params_buffer = []

def read_data(file_path):
    wavenumbers = []
    intensities = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines[1:]:  # 跳过第一行
            parts = line.strip().split()
            if len(parts) > 1:
                wavenumbers.append(float(parts[0]))
                intensities.append(float(parts[1]))
    
    return wavenumbers, intensities

def plot_data(wavenumbers, intensities, label_type="Raman", title="Wavenumber vs Intensity", color="blue", output_folder="."):
    plt.figure(figsize=(10, 6))
    
    if label_type == "Raman":
        xlabel = "Raman shift (cm-1)"
        ylabel = "Intensity"
    elif label_type == "Energy":
        xlabel = "Energy (eV)"
        ylabel = "Intensity"
    else:
        raise ValueError("Invalid label_type. Choose 'Raman' or 'Energy'.")
    
    plt.plot(wavenumbers, intensities, color=color, label="Original Data")
    
    for curve in curve_params_buffer:
        curve_type, params = curve['type'], curve['params']
        x = np.array(wavenumbers)
        params = list(map(float, params.split(',')))
        
        if curve_type == "Gaussian":
            y = gaussian(x, *params)
        elif curve_type == "Lorentz":
            y = lorentzian(x, *params)
        elif curve_type == "Voigt":
            y = voigt(x, *params)
        else:
            raise ValueError("Invalid curve type. Choose 'Gaussian', 'Lorentz', or 'Voigt'.")
        
        plt.plot(x, y, label=f"{curve_type} Curve", linestyle='--')
        plt.fill_between(x, y, alpha=0.3)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(output_folder, f"{title}.png")
    plt.savefig(output_path)
    plt.close()

def select_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
    file_list.delete(0, tk.END)
    for file_path in file_paths:
        file_list.insert(tk.END, file_path)

def select_output_folder():
    folder_path = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, folder_path)

def generate_plots():
    files = file_list.get(0, tk.END)
    output_folder = output_folder_entry.get()
    label_type = label_type_var.get()
    color = color_entry.get()
    
    if not files:
        messagebox.showerror("Error", "No files selected")
        return
    
    if not output_folder:
        messagebox.showerror("Error", "No output folder selected")
        return
    
    for file in files:
        wavenumbers, intensities = read_data(file)
        file_name = os.path.basename(file)
        title = f"{file_name} - {label_type}"
        plot_data(wavenumbers, intensities, label_type=label_type, title=title, color=color, output_folder=output_folder)
    
    messagebox.showinfo("Success", "Plots generated successfully")

def open_curve_window():
    curve_window = tk.Toplevel(root)
    curve_window.title("Add Curve")

    tk.Label(curve_window, text="Select file:").grid(row=0, column=0, padx=10, pady=5)
    file_var = tk.StringVar(value=file_list.get(0))
    tk.OptionMenu(curve_window, file_var, *file_list.get(0, tk.END)).grid(row=0, column=1, padx=10, pady=5)

    tk.Label(curve_window, text="Curve type:").grid(row=1, column=0, padx=10, pady=5)
    curve_type_var = tk.StringVar(value="Gaussian")
    tk.OptionMenu(curve_window, curve_type_var, "Gaussian", "Lorentz", "Voigt").grid(row=1, column=1, padx=10, pady=5)

    tk.Label(curve_window, text="Parameters:").grid(row=2, column=0, padx=10, pady=5)
    params_entry = tk.Entry(curve_window, width=50)
    params_entry.grid(row=2, column=1, padx=10, pady=5)

    tk.Button(curve_window, text="Add Curve", command=lambda: add_curve_to_buffer(file_var.get(), curve_type_var.get(), params_entry.get(), curve_listbox), bg="lightblue").grid(row=3, column=0, columnspan=2, padx=10, pady=20)

    # 添加列表框显示缓冲区中的曲线参数
    curve_listbox = tk.Listbox(curve_window, width=80, height=10)
    curve_listbox.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

    # 添加修改和删除按钮
    tk.Button(curve_window, text="Modify Curve", command=lambda: modify_curve(curve_listbox), bg="lightyellow").grid(row=5, column=0, padx=10, pady=5)
    tk.Button(curve_window, text="Delete Curve", command=lambda: delete_curve(curve_listbox), bg="lightcoral").grid(row=5, column=1, padx=10, pady=5)

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))

def lorentzian(x, amp, cen, wid):
    return amp * wid**2 / ((x-cen)**2 + wid**2)

def voigt(x, amp, cen, sigma, gamma):
    z = ((x-cen) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def add_curve_to_buffer(file_path, curve_type, params, curve_listbox):
    curve_params_buffer.append({'file': file_path, 'type': curve_type, 'params': params})
    curve_listbox.insert(tk.END, f"{curve_type} Curve: {params}")
    messagebox.showinfo("Success", f"{curve_type} curve parameters added to buffer")

def modify_curve(curve_listbox):
    selected_index = curve_listbox.curselection()
    if not selected_index:
        messagebox.showerror("Error", "No curve selected")
        return
    
    selected_index = selected_index[0]
    curve = curve_params_buffer[selected_index]
    
    modify_window = tk.Toplevel(root)
    modify_window.title("Modify Curve")

    tk.Label(modify_window, text="Curve type:").grid(row=0, column=0, padx=10, pady=5)
    curve_type_var = tk.StringVar(value=curve['type'])
    tk.OptionMenu(modify_window, curve_type_var, "Gaussian", "Lorentz", "Voigt").grid(row=0, column=1, padx=10, pady=5)

    tk.Label(modify_window, text="Parameters:").grid(row=1, column=0, padx=10, pady=5)
    params_entry = tk.Entry(modify_window, width=50)
    params_entry.insert(0, curve['params'])
    params_entry.grid(row=1, column=1, padx=10, pady=5)

    def save_modifications():
        curve_params_buffer[selected_index] = {'file': curve['file'], 'type': curve_type_var.get(), 'params': params_entry.get()}
        curve_listbox.delete(selected_index)
        curve_listbox.insert(selected_index, f"{curve_type_var.get()} Curve: {params_entry.get()}")
        modify_window.destroy()
        messagebox.showinfo("Success", "Curve parameters modified successfully")

    tk.Button(modify_window, text="Save", command=save_modifications, bg="lightblue").grid(row=2, column=0, columnspan=2, padx=10, pady=20)

def delete_curve(curve_listbox):
    selected_index = curve_listbox.curselection()
    if not selected_index:
        messagebox.showerror("Error", "No curve selected")
        return
    
    selected_index = selected_index[0]
    del curve_params_buffer[selected_index]
    curve_listbox.delete(selected_index)
    messagebox.showinfo("Success", "Curve parameters deleted successfully")

# 创建主窗口
root = tk.Tk()
root.title("Spectrum Plot Generator")

# 文件选择
tk.Label(root, text="Select txt files:").grid(row=0, column=0, padx=10, pady=5)
tk.Button(root, text="Select Files", command=select_files, bg="lightgreen").grid(row=0, column=1, padx=10, pady=5)
file_list = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50)
file_list.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

# 输出文件夹选择
tk.Label(root, text="Select output folder:").grid(row=2, column=0, padx=10, pady=5)
output_folder_entry = tk.Entry(root, width=50)
output_folder_entry.grid(row=3, column=0, padx=10, pady=5)
tk.Button(root, text="Select Folder", command=select_output_folder, bg="lightgreen").grid(row=3, column=1, padx=10, pady=5)

# 绘图参数
tk.Label(root, text="Label type:").grid(row=4, column=0, padx=10, pady=5)
label_type_var = tk.StringVar(value="Raman")
tk.OptionMenu(root, label_type_var, "Raman", "PL").grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Plot title:").grid(row=5, column=0, padx=10, pady=5)
title_entry = tk.Entry(root, width=50)
title_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Color:").grid(row=6, column=0, padx=10, pady=5)
color_entry = tk.Entry(root, width=50)
color_entry.grid(row=6, column=1, padx=10, pady=5)

# 生成图像按钮
tk.Button(root, text="Generate Plots", command=generate_plots, bg="lightblue").grid(row=7, column=0, columnspan=2, padx=10, pady=20)

# 在生成图像按钮下方添加一个按钮，用于打开子窗口
tk.Button(root, text="Add Curve", command=open_curve_window, bg="lightblue").grid(row=8, column=0, columnspan=2, padx=10, pady=20)

# 运行主循环
root.mainloop()