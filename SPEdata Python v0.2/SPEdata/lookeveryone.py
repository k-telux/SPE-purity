import matplotlib.pyplot as plt
import numpy as np

# 从txt中读取数据
def load_data(file_name):
    with open(file_name, 'r') as file:
        data = np.loadtxt(file)
    return data

# 读取数据
file_name = './SPE data/S1_BiTri/Raman/bilaer_3400mk_532.txt'  # 替换为你的文件路径
data = load_data(file_name)
x_data, y_data = data[:, 0], data[:, 1]
# 限制x值范围
x_min, x_max = 0, 600  # 设置你想要的范围
mask = (x_data >= x_min) & (x_data <= x_max)
x_data_filtered, y_data_filtered = x_data[mask], y_data[mask]
# 分段x值
segment1_mask = (x_data_filtered >= 230) & (x_data_filtered <= 270)
segment2_mask = (x_data_filtered >= 290) & (x_data_filtered <= 320)
#segment3_mask = (x_data_filtered >= 1250) & (x_data_filtered <= 1700)
#segment4_mask = (x_data_filtered >= 750) & (x_data_filtered <= 7500)

x_segment1, y_segment1 = x_data_filtered[segment1_mask], y_data_filtered[segment1_mask]
x_segment2, y_segment2 = x_data_filtered[segment2_mask], y_data_filtered[segment2_mask]
#x_segment3, y_segment3 = x_data_filtered[segment3_mask], y_data_filtered[segment3_mask]
#x_segment4, y_segment4 = x_data_filtered[segment2_mask], y_data_filtered[segment2_mask]

# 绘制散点图
'''plt.scatter(data[:, 0], data[:, 1], marker='.', color='blue')
plt.title('bilaer_3400mk_532')
plt.xlabel('Ramanshift(cm-1)')
plt.ylabel('Counts')
plt.show()'''
# plt.scatter(x_data_filtered, y_data_filtered, marker='.', color='blue', label='Data')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_segment1, y_segment1, marker='.',color='red', label='Segment 1 Lorentz Fit')
plt.title('Fitting Data')
plt.xlabel('Ramanshift(cm-1)')
plt.ylabel('Counts')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(x_segment2, y_segment2, marker='.',color='green', label='Segment 2 Lorentz Fit')
#plt.scatter(x_segment3, y_segment3, marker='.',color='blue', label='Segment 3 Lorentz Fit')
plt.title('Fitting Data')
plt.xlabel('Ramanshift(cm-1)')
plt.ylabel('Counts')
plt.legend()

plt.tight_layout()
plt.show()
