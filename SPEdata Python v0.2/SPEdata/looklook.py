import matplotlib.pyplot as plt
import numpy as np

# 从txt中读取数据
def load_data(file_name):
    with open(file_name, 'r') as file:
        data = np.loadtxt(file,delimiter=',\t')
    return data

# 读取数据
data1 = load_data('E:\Research in Rice\SPEdata Python v0.2\Figure S8\WL_BL_532.txt')
data2 = load_data('E:\Research in Rice\SPEdata Python v0.2\Figure S8\WL_BL_633.txt')
data3 = load_data('E:\Research in Rice\SPEdata Python v0.2\Figure S8\WL_BL_785.txt')

x_data1, y_data1 = data1[:, 0], data1[:, 1]
x_data2, y_data2 = data2[:, 0], data2[:, 1]
x_data3, y_data3 = data3[:, 0], data3[:, 1]

# 限制x值范围
x_min, x_max = 1.52, 1.6  # 设置你想要的范围
mask = (x_data1 >= x_min) & (x_data1 <= x_max)
x_data1, y_data1 = x_data1[mask], y_data1[mask]
mask = (x_data2 >= x_min) & (x_data2 <= x_max)
x_data2, y_data2 = x_data2[mask], y_data2[mask]
mask = (x_data3 >= x_min) & (x_data3 <= x_max)
x_data3, y_data3 = x_data3[mask], y_data3[mask]
# 绘制散点图
plt.plot(x_data1, y_data1, color='#F0A19A', label='532nm')
plt.plot(x_data2, y_data2, color='#7C7CBA', label='633nm')
plt.plot(x_data3, y_data3, color='#3FA0C0', label='785nm')

plt.plot()

plt.xlabel('Energy/eV')
plt.ylabel('Intensity(a.u.)')
plt.legend()
plt.show()
