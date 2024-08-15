import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
from sklearn.metrics import mean_squared_error, r2_score

# 洛伦兹函数
def lorentz(x, a, b, c):
    return a / ((x - b)**2 + c**2)
# 高斯函数
def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 / (2 * c**2))

# Voigt函数（高斯和洛伦兹的组合）
def voigt(x, a, b, c, d):
    return a * np.real(wofz((x - b + 1j*d) / (c * np.sqrt(2))))
#double function for mult peaks
def double_lorentz(x,a1,b1,c1,a2,b2,c2):
    return a1 / ((x - b1)**2 + c1**2) + a2 / ((x - b2)**2 + c2**2)

def double_gaussian(x,a1,b1,c1,a2,b2,c2):
    return a1 * np.exp(-(x - b1)**2 / (2 * c1**2)) + a2 * np.exp(-(x - b2)**2 / (2 * c2**2))

def double_voigt(x,a1,b1,c1,d1,a2,b2,c2,d2):
    return a1 * np.real(wofz((x - b1 + 1j*d1) / (c1 * np.sqrt(2)))) + a2 * np.real(wofz((x - b2 + 1j*d2) / (c2 * np.sqrt(2))))
def fitfit(functiontype, xdata, ydata, xmin, xmax):
    segment1_mask = (xdata >= xmin) & (xdata <= xmax)
    x_segment1, y_segment1 = xdata[segment1_mask], ydata[segment1_mask]
    # 拟合线形
    params1, cov1 = curve_fit(functiontype, x_segment1, y_segment1)
    param_errors1 = np.sqrt(np.diag(cov1))
    # 计算拟合评价指标
    y_1 = functiontype(x_segment1, *params1)
    xplay = np.arange(xmin,xmax,0.1)
    yplay = functiontype(xplay, *params1)
    rmse_1 = np.sqrt(mean_squared_error(y_segment1, y_1))
    r2_1 = r2_score(y_segment1, y_1)
    for i, params1 in enumerate(params1):
        print(f'Parameter1 {i + 1}: {params1:.2f} ± {param_errors1[i]:.2f}')
    print(f"Segment 1 RMSE：{rmse_1:.4f}, R²：{r2_1:.4f}")
    plt.scatter(x_segment1, y_segment1, marker='.', color='blue', label='Data')

    plt.plot(xplay, yplay, color='red', label='Segment 1 Voigt Fit')
    plt.title('Segment 1 Fitting Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    return params1, param_errors1
def bestfit(xdata, ydata, xmin, xmax):
    segment1_mask = (xdata >= xmin) & (xdata <= xmax)
    x_segment1, y_segment1 = xdata[segment1_mask], ydata[segment1_mask]
    # 拟合线形
    params1, cov1 = curve_fit(lorentz, x_segment1, y_segment1)
    param_errors1 = np.sqrt(np.diag(cov1))
    # 计算拟合评价指标
    y_1 = lorentz(x_segment1, *params1)
    rmse_1 = np.sqrt(mean_squared_error(y_segment1, y_1))
    r2_1 = r2_score(y_segment1, y_1)
    # 拟合线形
    params2, cov2 = curve_fit(gaussian, x_segment1, y_segment1)
    param_errors2 = np.sqrt(np.diag(cov2))
    # 计算拟合评价指标
    y_2 = gaussian(x_segment1, *params2)
    rmse_2 = np.sqrt(mean_squared_error(y_segment1, y_2))
    r2_2 = r2_score(y_segment1, y_2)
    # 拟合线形
    try:
        params3, cov3 = curve_fit(voigt, x_segment1, y_segment1)
    except RuntimeError:
        params3, cov3 = [1,1,1,1],[]
    param_errors3 = np.sqrt(np.diag(cov3))
    # 计算拟合评价指标
    y_3 = voigt(x_segment1, *params3)
    rmse_3 = np.sqrt(mean_squared_error(y_segment1, y_3))
    r2_3 = r2_score(y_segment1, y_3)
    try:
        params4, cov4 = curve_fit(double_lorentz, x_segment1, y_segment1)
    except RuntimeError:
        params4, cov4 = [1,1,1,1,1,1],[]
    param_errors4 = np.sqrt(np.diag(cov4))
    # 计算拟合评价指标
    y_4 = double_lorentz(x_segment1, *params4)
    rmse_4 = np.sqrt(mean_squared_error(y_segment1, y_4))
    r2_4 = r2_score(y_segment1, y_4)
    try:
        params5, cov5 = curve_fit(double_gaussian, x_segment1, y_segment1)
    except RuntimeError:
        params5, cov5 = [1, 1, 1, 1, 1, 1], []
    param_errors5 = np.sqrt(np.diag(cov5))
    # 计算拟合评价指标
    y_5 = double_gaussian(x_segment1, *params5)
    rmse_5 = np.sqrt(mean_squared_error(y_segment1, y_5))
    r2_5 = r2_score(y_segment1, y_5)
    try:
        params6, cov6 = curve_fit(double_voigt, x_segment1, y_segment1)
    except RuntimeError:
        params6, cov6 = [1, 1, 1, 1, 1, 1,1,1], []
    param_errors6 = np.sqrt(np.diag(cov6))
    # 计算拟合评价指标
    y_6 = double_voigt(x_segment1, *params6)
    rmse_6 = np.sqrt(mean_squared_error(y_segment1, y_6))
    r2_6 = r2_score(y_segment1, y_6)
    parameter = [params1,params2,params3,params4,params5,params6]
    parametererror = [param_errors1, param_errors2, param_errors3, param_errors4, param_errors5, param_errors6]
    r2 = np.array([r2_1,r2_2,r2_3,r2_4,r2_5,r2_6])
    max_index = r2.argmax()
    func = [lorentz,gaussian,voigt,double_lorentz,double_gaussian,double_voigt]
    xplay = np.arange(xmin, xmax, 0.1)
    yplay = func[max_index](xplay, *parameter[max_index])
    if max_index ==3:
        yplay1 = func[max_index-3](xplay, parameter[max_index][0],parameter[max_index][1],parameter[max_index][2])
        yplay2 = func[max_index-3](xplay, parameter[max_index][3],parameter[max_index][4],parameter[max_index][5])
    parametercopy = parameter
    for i, parameter[max_index] in enumerate(params1):
        print(f'Parameter1 {i + 1}: {parameter[max_index]:.2f} ± { parametererror[max_index][i]:.2f}')
    print(f"Segment R²：{r2[max_index]:.4f}")
    print(max_index+1)
    plt.scatter(x_segment1, y_segment1, marker='.', color='blue', label='Data')

    plt.plot(xplay, yplay, color='red', label='Segment 1 Voigt Fit')
    if max_index == 3:
        plt.plot(xplay,yplay1,color='green')
        plt.plot(xplay,yplay2,color='orange')
    plt.title('Segment 1 Fitting Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    return parameter
