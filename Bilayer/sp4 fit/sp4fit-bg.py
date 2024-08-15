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
file = 'bilayer_sp4.txt'
data = np.loadtxt(file, delimiter=',')
x_values = data[:, 0]
y_values = data[:, 1]
y_corrected=y_values-gaussian(x_values,1264,1.51548, 0.05759)-gaussian(x_values,1000,1.52902, 0.05121)-gaussian(x_values,1640 ,1.56422,0.02042)
plt.plot(x_values,y_corrected , label='Data')
#np.savetxt('corrected_data_sp4.txt', np.column_stack((x_values, y_corrected)), delimiter=',', header='Energy (eV), Intensity (a.u.)', comments='')
plt.plot(x_values, y_values,'r')
'''
plot_lines(x_values, 'lorentz', 294, 1.44591, 0.008, 0, 0)
plot_lines(x_values, 'lorentz', 503, 1.46338, 0.035, 0, 0)
plot_lines(x_values, 'lorentz', 1716, 1.48, 0.012, 0, 0)
plot_lines(x_values, 'lorentz', 1847, 1.50032, 0.006, 0, 0)

plot_lines(x_values, 'gaussian', 2130, 1.51702, 0.005, 0, 0)
plot_lines(x_values, 'gaussian', 2244, 1.52416, 0.016, 0, 0)
plot_lines(x_values, 'gaussian', 2378, 1.53773, 0.008, 0, 0)
plot_lines(x_values, 'gaussian', 4772, 1.55094, 0.008, 0, 0)

plot_lines(x_values, 'lorentz', 3068, 1.56399, 0.005, 0, 0)
plot_lines(x_values, 'lorentz', 2000, 1.56891, 0.005, 0, 0)
plot_lines(x_values, 'lorentz', 1431, 1.57475, 0.005, 0, 0)
plot_lines(x_values, 'lorentz', 894, 1.57898, 0.001, 0, 0)

plot_lines(x_values, 'gaussian', 1058, 1.64782, 0.001, 0, 0)
'''
# main peak
plot_lines(x_values, 'lorentz', 9633, 1.51557, 0.0029, 0, 0)
plot_lines(x_values, 'lorentz', 9488, 1.55359, 0.0007, 0, 0)
plot_lines(x_values, 'lorentz', 13756, 1.57041, 0.0012, 0, 0)

plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'{file}')
plt.show()