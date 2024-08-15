import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

# 定义二阶自相干函数 g^(2)(tau)
def g2_function(tau, g2_0, tau_c):
    return 1-(1-g2_0) * np.exp(-np.abs(tau) / tau_c)

# 读取多个文件的数据
file_paths = glob.glob('E:\Research in Rice\SPEdata Python v0.2\Figure S9\*.txt')
data_sets = []
for file_path in file_paths:
    data = np.loadtxt(file_path,delimiter=',\t')
    tau = data[:, 0]
    g2_data = data[:, 1]
    data_sets.append((tau, g2_data))

# 拟合函数并绘制结果
fig, axes = plt.subplots(1,len(data_sets),figsize=(18, 4))

for i, (tau, g2_data) in enumerate(data_sets):
    popt, pcov = curve_fit(g2_function, tau, g2_data)
    g2_0_fit, tau_c_fit = popt
    axes[i].scatter(tau, g2_data, s=5, label=f'BL Data set {i + 1}',color=(88/255, 97/255, 172/255))
    axes[i].plot(tau, g2_function(tau, *popt), label=f'Fitted g2 {i + 1}', color=(238/255, 202/255, 64/255))
    axes[i].set_xlabel('$t$')
    axes[i].set_ylabel('$g^{(2)}(t)$')
    axes[i].legend()
    axes[i].text(0.6, 0.25, f'Fitted $g^{(2)}(0)$: {g2_0_fit:.3f}\nFitted $t_c$: {tau_c_fit:.2f}',
                 transform=axes[i].transAxes, fontsize=12, verticalalignment='top')
    print(g2_0_fit)


plt.show()


