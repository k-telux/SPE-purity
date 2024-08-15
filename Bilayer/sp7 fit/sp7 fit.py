import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz

def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-2.773*((x - mean) ** 2) / ( std_dev ** 2))

def lorentzian(x, amplitude, mean, gamma):
    return amplitude * (gamma / (1 + ((x - mean) / gamma) ** 2))

def voigt(x, amplitude, mean, gamma, sigma):
    z = (x - mean) + 1j * gamma
    return amplitude * np.real(wofz(z) / z) * np.exp(-(sigma ** 2) * (z.imag ** 2))

def plot_lines(x_values, line_type, amplitude, mean, gamma, sigma,y0):
    if line_type == 'gaussian':    
        line = gaussian(x_values, amplitude, mean, gamma)
        plt.plot(x_values, line+y0, label=f'{line_type} - Amplitude: {amplitude}, Mean: {mean}, Gamma: {gamma}')
        plt.legend()
    elif line_type == 'lorentzian':
        line = lorentzian(x_values, amplitude, mean, gamma)
        plt.plot(x_values, line+y0, label=f'{line_type} - Amplitude: {amplitude}, Mean: {mean}, Gamma: {gamma}')
        plt.legend()
    elif line_type == 'voigt':
        line_function = voigt
        line = voigt(x_values, amplitude, mean, gamma,sigma)
        plt.plot(x_values, line+y0, label=f'{line_type} - Amplitude: {amplitude}, Mean: {mean}, Gamma: {gamma}, Sigma: {sigma})')
        plt.legend()
    else:
        print("Invalid line type. Please choose 'gaussian', 'lorentzian', or 'voigt'.")
        return

# Example usage
file = 'bilayer_sp3.txt'
data = np.loadtxt(file, delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]

plt.plot(x_values, y_values, label='Data')

# Add Gaussian line
plot_lines(x_values, 'gaussian', 1545, 1.482, 0.0140, 0, 597)
plot_lines(x_values, 'gaussian', 2527, 1.499, 0.0085, 0, 597)
plot_lines(x_values, 'gaussian', 4243, 1.521, 0.0217, 0, 597)
plot_lines(x_values, 'gaussian', 1544.5, 1.482, 0.014, 0, 597)
plot_lines(x_values, 'gaussian', 2527, 1.499, 0.0085, 0, 597)
plot_lines(x_values, 'gaussian', 4243, 1.521, 0.0217, 0, 597)

# Add Lorentzian line
plot_lines(x_values, 'lorentzian', 50, 3.0, 0.2,0,1)

# Add Voigt line
plot_lines(x_values, 'voigt', 200, 2.8, 0.1, 0.05,1)

plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'{file}')
plt.show()