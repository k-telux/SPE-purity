import matplotlib.pyplot as plt
import numpy as np

# 从txt中读取数据
def load_data(file_name):
    with open(file_name, 'r') as file:
        data = np.loadtxt(file, delimiter=',\t')
    return data

# 读取数据
data1 = load_data('E:\Research in Rice\SPEdata Python v0.2\FigureS10\POL_ML_2_PARA.txt')
data2 = load_data('E:\Research in Rice\SPEdata Python v0.2\FigureS10\POL_ML_2_CROS.txt')

x_data1, y_data1 = data1[:, 0], data1[:, 1]
x_data2, y_data2 = data2[:, 0], data2[:, 1]
# 限制x值范围
x_min, x_max = 1.58, 1.75 # 设置你想要的范围
mask = (x_data1 >= x_min) & (x_data1 <= x_max)
x_data1, y_data1 = x_data1[mask], y_data1[mask]
mask = (x_data2 >= x_min) & (x_data2 <= x_max)
x_data2, y_data2= x_data2[mask], y_data2[mask]
# 绘制散点图
plt.plot(x_data1, y_data1, color=(240/255, 161/255, 154/255), label=r'$\sigma^{+} \sigma^{+}$')
plt.plot(x_data2, y_data2, color=(124/255, 124/255, 186/255), label=r'$\sigma^{+} \sigma^{-}$')

plt.legend()
plt.xlabel('Energy/eV')
plt.ylabel('Intensity(a.u.)')
plt.show()