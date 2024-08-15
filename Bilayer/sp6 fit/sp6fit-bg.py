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
file = 'bilayer_sp6.txt'
data = np.loadtxt(file, delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]
y_corrected=y_values-gaussian(x_values,1500 ,1.55308, 0.046)
plt.plot(x_values,y_corrected , label='Data')
np.savetxt('no bd_sp6.txt', np.column_stack((x_values, y_corrected)), delimiter=',', header='Energy (eV), Intensity (a.u.)', comments='')
plt.plot(x_values, y_values,'r')

plot_lines(x_values, 'lorentz', 4356, 1.52877, 0.00216, 0, 0)
plot_lines(x_values, 'lorentz', 4760, 1.53016, 0.00207, 0, 0)
plot_lines(x_values, 'lorentz', 4090, 1.53453, 0.00114, 0, 0)
plot_lines(x_values, 'lorentz', 2411, 1.54140, 0.00138, 0, 0)
plot_lines(x_values, 'lorentz', 2588, 1.55272, 0.00476, 0, 0)
plot_lines(x_values, 'lorentz', 4485, 1.55502, 0.00113, 0, 0)
plot_lines(x_values, 'gaussian', 1203, 1.58636, 0.00313,0, 0)
plot_lines(x_values, 'gaussian', 1172, 1.59851, 0.00309,0, 0)
# main peak
#plot_lines(x_values, 'gaussian', 2188, 1.57048,0.01131,0,0)
plot_lines(x_values, 'lorentz', 8086, 1.57741,0.00184,0,0)
plot_lines(x_values, 'lorentz', 4472, 1.57830,0.00039,0,0)

plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'{file}')
plt.show()
