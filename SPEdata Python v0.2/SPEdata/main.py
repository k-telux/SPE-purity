import matplotlib.pyplot as plt
import numpy as np
import fitfunction as ff
from scipy.optimize import curve_fit
from scipy.special import wofz
from sklearn.metrics import mean_squared_error, r2_score

# 从txt中读取数据
def load_data(file_name):
    with open(file_name, 'r') as file:
        data = np.loadtxt(file)
    return data

# 读取数据
file_name = './SPE data/S1_BiTri/photoluminenscence/bi_sp2_3400mk_532.txt'  # 替换为你的文件路径
data = load_data(file_name)
x_data, y_data = data[:, 0], data[:, 1]

# 洛伦兹函数
def lorentz(x, a, b, c):
    return a / ((x - b)**2 + c**2)
# 高斯函数
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Voigt函数（高斯和洛伦兹的组合）
def voigt(x, a, b, c, d):
    return a * np.real(wofz((x - b + 1j*d) / (c * np.sqrt(2))))
'''
# 拟合线形
params1,cov1 = curve_fit(voigt, x_segment1, y_segment1)
params2,cov2 = curve_fit(lorentz, x_segment2, y_segment2)

param_errors1 = np.sqrt(np.diag(cov1))
param_errors2 = np.sqrt(np.diag(cov2))

# 计算拟合评价指标
y_1 = voigt(x_segment1, *params1)
y_2 = lorentz(x_segment2, *params2)

rmse_1 = np.sqrt(mean_squared_error(y_segment1, y_1))
rmse_2 = np.sqrt(mean_squared_error(y_segment2, y_2))

r2_1 = r2_score(y_segment1, y_1)
r2_2 = r2_score(y_segment2, y_2)

for i, params1 in enumerate(params1):
    print(f'Parameter1 {i+1}: {params1:.2f} ± {param_errors1[i]:.2f}')
for i, params2 in enumerate(params2):
    print(f'Parameter2 {i+1}: {params2:.2f} ± {param_errors2[i]:.2f}')
print(f"Segment 1 RMSE：{rmse_1:.4f}, R²：{r2_1:.4f}")
print(f"Segment 2 RMSE：{rmse_2:.4f}, R²：{r2_2:.4f}")
'''
#para,paraerr = ff.fitfit(gaussian,x_data,y_data,230,270)
ff.bestfit(x_data,y_data,775,820)
# 绘制拟合曲线
''' plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(x_segment1, y_segment1, marker='.', color='blue', label='Data')
plt.plot(x_segment1, y_1, color='red', label='Segment 1 Voigt Fit')
plt.title('Segment 1 Fitting Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_segment2, y_segment2, marker='.', color='blue', label='Data')
plt.plot(x_segment2, y_2, color='green', label='Segment 2 Voigt Fit')
plt.title('Segment 2 Fitting Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

plt.tight_layout()
plt.show()
'''
