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
file = '4k-4PL.txt'
data = np.loadtxt(file, delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]

filtered_x_values = x_values[(x_values > 0) & (x_values < 500)]
filtered_y_values = y_values[(x_values > 0) & (x_values < 500)]

y_corrected=filtered_y_values-gaussian(filtered_x_values, 6157, 1.637, 0.05)
plt.plot(filtered_x_values,y_corrected , label='Data')
np.savetxt('corrected_data.txt', np.column_stack((filtered_x_values, y_corrected)), delimiter=',', header='Energy (eV), Intensity (a.u.)', comments='')
plt.plot(filtered_x_values, filtered_y_values,'r')

plot_lines(filtered_x_values, "lorentz", 2596, 1.613, 0.009,0,0)
plot_lines(filtered_x_values, "lorentz", 15321, 1.637, 0.003,0,0)
plot_lines(filtered_x_values, "lorentz", 5737, 1.647, 0.012,0,0)
plot_lines(filtered_x_values, "lorentz", 12421, 1.664, 0.019,0,0)
plot_lines(filtered_x_values, "lorentz", 6778, 1.681, 0.028,0,0)

plot_lines(filtered_x_values, "lorentz", 9990, 1.718, 0.016,0,0)
plot_lines(filtered_x_values, "gaussian", 2581, 1.736, 0.009,0,0)

plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'{file}')
plt.show()