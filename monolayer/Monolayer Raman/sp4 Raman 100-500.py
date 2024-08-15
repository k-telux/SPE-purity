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
file = '4k-4Raman.txt'
data = np.loadtxt(file, delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]
# Filter x_values to only include values between 100 and 500
filtered_x_values = x_values[(x_values > 100) & (x_values < 500)]
filtered_y_values = y_values[(x_values > 100) & (x_values < 500)]

y_corrected=filtered_y_values-gaussian(filtered_x_values, 100, 264.2353, 422)
plt.plot(filtered_x_values,y_corrected , label='Data')
np.savetxt('corrected_data1.txt', np.column_stack((filtered_x_values, y_corrected)), delimiter=',', header='Energy (eV), Intensity (a.u.)', comments='')
plt.plot(filtered_x_values, filtered_y_values,'r')

plot_lines(filtered_x_values, "lorentz", 141, 223.29, 4.51, 0,0)
plot_lines(filtered_x_values, "lorentz", 307, 240.34, 3.60, 0,0)
plot_lines(filtered_x_values, "lorentz", 729, 252.21, 5.39, 0,0)
plot_lines(filtered_x_values, "lorentz", 457, 264.58, 2.25, 0,0)


plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'{file}')
plt.show()