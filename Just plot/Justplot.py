# Python
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import os

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
    
    plt.plot(wavenumbers, intensities, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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

# 创建主窗口
root = tk.Tk()
root.title("Spectrum Plot Generator")

# 文件选择
tk.Label(root, text="Select txt files:").grid(row=0, column=0, padx=10, pady=5)
tk.Button(root, text="Select Files", command=select_files).grid(row=0, column=1, padx=10, pady=5)
file_list = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50)
file_list.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

# 输出文件夹选择
tk.Label(root, text="Select output folder:").grid(row=2, column=0, padx=10, pady=5)
output_folder_entry = tk.Entry(root, width=50)
output_folder_entry.grid(row=3, column=0, padx=10, pady=5)
tk.Button(root, text="Select Folder", command=select_output_folder).grid(row=3, column=1, padx=10, pady=5)

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
tk.Button(root, text="Generate Plots", command=generate_plots).grid(row=7, column=0, columnspan=2, padx=10, pady=20)

# 运行主循环
root.mainloop()