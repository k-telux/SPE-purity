import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import wofz
# Read data from the .txt file

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

def plot_data(kind,allornot,savefig,shadow,line_color='blue',shadow_color='gray'):
    file_list = sorted([file for file in os.listdir() if file.endswith('.txt')], key=lambda file: float(file.split('_')[2].split('.')[0]))
    fig, ax = plt.subplots()
    if kind == 'PL':
        xlabel='Energy (eV)'
        ylabel='PL Intensity (a.u.)'
        title='PL Intensity'

    elif kind == 'Raman':
        xlabel='Raman Shift (cm-1)'
        ylabel='Intensity (a.u.)'
        title='Raman Intensity'

    elif kind == 'g2':
        xlabel='Time (ns)'
        ylabel='g$^{2}(t)$'
        title='g$^{2}(t)$'
    
    if allornot == 'all':
        for file in file_list:
            data = np.loadtxt(file, delimiter=',')
            x_values = data[:, 0]
            y_values = data[:, 1]
            # Extract the spot sort number from the file name
            spot_sort_number = file.split('_')[1].split('.')[0]
            ax.plot(x_values, y_values, label=f'{spot_sort_number}')
            if shadow==True:
                ax.fill_between(x_values, y_values, alpha=0.5)
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        plt.title(f'{title}')
        plt.show()
        if savefig==True:
            plt.savefig(f'{kind}_all_data_plot.svg', format='svg')

    elif allornot == 'single':
        for file in file_list:
            data = np.loadtxt(file, delimiter=',')
            x_values = data[:, 0]
            y_values = data[:, 1]
            # Extract the spot sort number from the file name
            spot_sort_number = file.split('_')[2].split('.')[0]
            plt.plot(x_values, y_values,color=line_color, label=f'{spot_sort_number}')
            if shadow==True:
                plt.fill_between(x_values, y_values, color=shadow_color, alpha=0.5)
            plt.xlabel(f'{xlabel}')
            plt.ylabel(f'{ylabel}')
            plt.title(f'{title}')
            plt.legend()
            if savefig==True:
                plt.savefig(f'{spot_sort_number}_data_plot.svg', format='svg')
            plt.show()
    
# Before using, rename the txt file into "xx_xx_number", the number will be a symbol to sort the plots
# 'PL','Raman','g2'
#  single means draw each txt file with a single plot; all means draw all files with a summary plot
plot_data( 'Raman', 'single', savefig=True, shadow=True,line_color='red',shadow_color='green')
