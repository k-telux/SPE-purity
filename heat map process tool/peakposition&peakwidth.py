import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.signal import find_peaks
def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))
def lorentzian(x, x0, gamma, A):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)
def voigt(x, amp, center, sigma, gamma):
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
def double_loretzian(x,x1,gamma,A1,amp,x2,A2):
    return lorentzian(x,x1,gamma,A1)+gaussian(x,amp,x2,A2)
#---------------------------------------------------------------------------------
'''
You can adjust the following parameters here, where 
filename is the filename of the data to be processed, 
threshold fwhm is the maximum width to be tolerated, 
and threshold peak is the range of tolerated peak positions, 
combined with error tolerance, which is the tolerance 
for error in order to filter out the points where no peaks 
actually exist that meet the requirements They are labeled white in the graph.
'''
filename = 'map_2_pl.txt'
Xlable = 'X Coordinate'
Ylable = 'Y Coordinate'
Title1 = 'Peak Position Heat Map'
Title2 = 'FWHM Heat Map'
threshold_fwhm = 0.1
threshold_peakmax = 2.04
threshold_peakmin = 1.9
funcname = voigt
error_tolerance = 0.1
#---------------------------------------------------------------------------------


# read data
data = pd.read_csv(filename, delimiter=r'\s+|\t', engine='python', header=None, names=['X', 'Y', 'Wave', 'Intensity'],skiprows=1)
# Get unique x and y coordinates
x_coords = data['X'].unique()
y_coords = data['Y'].unique()

# Initialize the result array
peak_map = np.zeros((len(y_coords), len(x_coords)))
fwhm_map = np.zeros((len(y_coords), len(x_coords)))

# Iterate over each point
for i, x in enumerate(x_coords):
    for j, y in enumerate(y_coords):
        subset = data[(data['X'] == x) & (data['Y'] == y)]
        if subset.empty:
            continue

        wave = subset['Wave'].values
        intensity = subset['Intensity'].values

        try:
            popt, pcov = curve_fit(funcname, wave, intensity)
            perr = np.sqrt(np.diag(pcov))
            if funcname == voigt:
                # 这里是筛选不合格数据的标准
                if perr[3]/popt[3] < error_tolerance:
                    peak_map[j, i] = popt[1]
                    fwhm_map[j, i] = 0.5346 * 2 * popt[3] + np.sqrt(0.2166 * (2 * popt[3])**2 + (2 * popt[2] * np.sqrt(2 * np.log(2)))**2)
                else:
                    peak_map[j, i] = np.nan
                    fwhm_map[j, i] = np.nan
            else:
                if perr[1]/popt[1] < error_tolerance:
                    peak_map[j, i] = popt[0]
                    fwhm_map[j, i] = popt[1]
                else:
                    peak_map[j, i] = np.nan
                    fwhm_map[j, i] = np.nan
        except RuntimeError:
            peak_map[j, i] = np.nan
            fwhm_map[j, i] = np.nan

# Sift out special points that exceed the threshold and replace them with NaN
fwhm_map[(fwhm_map > threshold_fwhm)|(fwhm_map < 0)] = np.nan
peak_map[(peak_map > threshold_peak1) | (peak_map < threshold_peak2)] = np.nan
# Drawing a heat map
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
c1 = ax[0].imshow(peak_map, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()), origin='lower',
                  aspect='auto', cmap='plasma')
ax[0].set_title(Title1)
ax[0].set_xlabel(Xlable)
ax[0].set_ylabel(Ylable)
fig.colorbar(c1, ax=ax[0])

c2 = ax[1].imshow(fwhm_map, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()), origin='lower',
                  aspect='auto', cmap='plasma')
ax[1].set_title(Title2)
ax[1].set_xlabel(Xlable)
ax[1].set_ylabel(Ylable)
fig.colorbar(c2, ax=ax[1])

plt.show()

