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
        line = voigt(x_values, amplitude, mean, gamma,sigma)
        plt.plot(x_values, line+y0, label=f'Amplitude: {amplitude}, Mean: {mean}, Gamma: {gamma}, Sigma: {sigma})')
        plt.legend()
    else:
        print("Invalid line type. Please choose 'gaussian', 'lorentz', or 'voigt'.")
        return

# Example usage
file = 'bilayer_sp8.txt'
data = np.loadtxt(file, delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]
y_corrected=y_values-gaussian(x_values, 3581, 1.53503, 0.03448)-gaussian(x_values, 2007, 1.55543, 0.00235)-gaussian(x_values,1500,1.5545,0.003)
plt.plot(x_values,y_corrected , label='Data')
#np.savetxt('no bd_sp8.txt', np.column_stack((x_values, y_corrected)), delimiter=',', header='Energy (eV), Intensity (a.u.)', comments='')
plt.plot(x_values, y_values,'r')

# side peaks
plot_lines(x_values, 'lorentz', 3717, 1.52837, 0.00091,0,0)
plot_lines(x_values, 'lorentz', 5590, 1.52952, 0.00079,0,0)
plot_lines(x_values, 'lorentz', 5476, 1.53274, 0.00071,0,0)
plot_lines(x_values, 'gaussian', 2230, 1.53665, 0.00264,0,0)
plot_lines(x_values, 'gaussian', 4469, 1.54089, 0.00445,0,0)

# main peak
plot_lines(x_values, 'lorentz', 13578, 1.53062, 0.00107,0,0)
plot_lines(x_values, 'lorentz', 13176, 1.53169, 0.00115,0,0)
plot_lines(x_values, 'lorentz', 8277, 1.55750, 0.00207,0,0)
#plot_lines(x_values, 'gaussian', 17189, 1.53065, 0.00081,0,0)
#plot_lines(x_values, 'gaussian', 16508, 1.53165, 0.00083,0,0)

plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'{file}')
plt.show()