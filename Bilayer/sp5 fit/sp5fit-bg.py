import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz

def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-2.773*((x - mean) ** 2) / ( std_dev ** 2))

def lorentz(x, amplitude, mean, gamma):
    return amplitude * (gamma*gamma / (gamma*gamma + 4*((x - mean) ** 2)))

def voigt(x, amp,x0, gamma, sigma):
    z = ((x - x0) + 1j * gamma) / (sigma * np.sqrt(2))
    result = amp * np.real(wofz(z))
    return np.clip(result, -np.inf, np.inf)

def plot_lines(x_values, line_type, amplitude, mean, gamma, sigma,y0):
    if line_type == 'gaussian':    
        line = gaussian(x_values, amplitude, mean, gamma)
        plt.plot(x_values, line+y0, label=f'Amplitude: {amplitude}, Mean: {mean}, Gamma: {gamma}')
        plt.legend()
    elif line_type == 'lorentz':
        line = lorentz(x_values, amplitude, mean, gamma)
        plt.plot(x_values, line+y0, label=f'Amplitude: {amplitude}, Mean: {mean}, Gamma: {gamma}')
        plt.legend()
    elif line_type == 'voigt':
        line_function = voigt
        line = voigt(x_values, amplitude, mean, gamma,sigma)
        plt.plot(x_values, line+y0, label=f'Amplitude: {amplitude}, Mean: {mean}, Gamma: {gamma}, Sigma: {sigma})')
        plt.legend()
    else:
        print("Invalid line type. Please choose 'gaussian', 'lorentz', or 'voigt'.")
        return

# Example usage
file = 'bilayer_sp5.txt'
data = np.loadtxt(file, delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]
y_corrected=y_values
plt.scatter(x_values,y_corrected , label='Data')
#np.savetxt('corrected_data_sp4.txt', np.column_stack((x_values, y_corrected)), delimiter=',', header='Energy (eV), Intensity (a.u.)', comments='')
plt.plot(x_values, y_values,'r')

plot_lines(x_values, 'lorentz', 12848, 1.53092, 0.0047, 0, 0)
plot_lines(x_values, 'lorentz', 7452, 1.53503, 0.0018, 0, 0)
plot_lines(x_values, 'lorentz', 11425, 1.53698, 0.0022, 0, 0)
plot_lines(x_values, 'lorentz', 11784, 1.53870,0.00362, 0, 0)
plot_lines(x_values, 'lorentz', 4475, 1.73204, 0.00786, 0, 0)
plot_lines(x_values, 'gaussian', 7600, 1.56874, 0.01312, 0, 0)
plot_lines(x_values, 'gaussian', 3922, 1.58346, 0.00360, 0, 0)
plot_lines(x_values, 'gaussian', 2603, 1.58765, 0.00543, 0, 0)
plot_lines(x_values, 'gaussian', 16143, 1.54663, 0.00360, 0, 0)
plot_lines(x_values, 'gaussian', 13143, 1.54957, 0.00541, 0, 0)
plot_lines(x_values, 'gaussian', 10411, 1.55709, 0.00543, 0, 0)

# main peak
plot_lines(x_values, 'gaussian', 17834, 1.54147, 0.00017, 0, 0)
plot_lines(x_values, 'lorentz', 14196, 1.53610, 0.00035, 0, 0)
plot_lines(x_values, 'lorentz', 13089, 1.72972, 0.00014, 0, 0)

plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'{file}')
plt.show()