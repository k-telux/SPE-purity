import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
def gaussian(x, cen, amp, wid):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2))

def lorentz(x, x0, gamma, A):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)

def voigt(x, center,amp, sigma, gamma):
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def double_lorentz(x,x1,x2,gamma1, A1,  gamma2, A2 ):
    return lorentz(x,x1,gamma1,A1)+lorentz(x,x2,gamma2,A2)
def double_gaussian(x, cen1,cen2, amp1,wid1,amp2,wid2 ):
    return gaussian(x,cen1,amp1,wid1)+gaussian(x,cen2,amp2,wid2)
def double_voigt(x,center1,center2,amp1,sigma1,gamma1,amp2,sigma2,gamma2):
    return voigt(x,center1,amp1,sigma1,gamma1)+voigt(x,center2,amp2,sigma2,gamma2)
def gaussianpluslorentz(x,cen,x0,amp,wid,gamma,A):
    return gaussian(x,cen,amp,wid)+lorentz(x,x0,gamma,A)
def trans(funcname):
    if funcname == 'lorentz':
        func = lorentz
    if funcname == 'gaussian':
        func = gaussian
    if funcname == 'voigt':
        func = voigt
    if funcname == 'double_lorentz':
        func = double_lorentz
    if funcname == 'double_gaussian':
        func = double_gaussian
    if funcname == 'double_voigt':
        func = double_voigt
    if funcname == 'gaussianpluslorentz':
        func = gaussianpluslorentz
    return func
def Itrans(funcname):
    if funcname == lorentz:
        func = 'lorentz'
    if funcname == gaussian:
        func = 'gaussian'
    if funcname == voigt:
        func = 'voigt'
    if funcname == double_lorentz:
        func = 'double_lorentz'
    if funcname == double_gaussian:
        func = 'double_gaussian'
    if funcname == double_voigt:
        func = 'double_voigt'
    if funcname == gaussianpluslorentz:
        func = 'gaussianpluslorentz'
    return func
def drawonepeakint(filename,Xlable,Ylable,Title1):
    # read data file
    data = np.loadtxt(filename)

    # Extract xy coordinates and spectral data
    x = data[:, 0]
    y = data[:, 1]
    wave = data[:, 2]
    intensity = data[:, 3]

    # Get unique x and y coordinates
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    # Create a two-dimensional array to store the integration results
    heatmap = np.zeros((len(unique_y), len(unique_x)))

    # Calculate the numerical integral for each spectrum and store it in the heatmap
    for i in range(len(unique_x)):
        for j in range(len(unique_y)):
            mask = (x == unique_x[i]) & (y == unique_y[j])
            if np.any(mask):
                heatmap[j, i] = np.trapz(intensity[mask], wave[mask])

    # Drawing heat maps
    plt.imshow(heatmap, extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()], origin='lower',
               aspect='auto', cmap='plasma')
    plt.colorbar()
    plt.xlabel(Xlable)
    plt.ylabel(Ylable)
    plt.title(Title1)
    plt.show()
    return 0


def drawonepeakposition(filename,Xlable,Ylable,Title1,threshold_peakmax,threshold_peakmin,fitfuncname,error_tolerance):
    # read data
    data = pd.read_csv(filename, delimiter=r'\s+|\t', engine='python', header=None,
                       names=['X', 'Y', 'Wave', 'Intensity'], skiprows=1)
    # Get unique x and y coordinates
    x_coords = data['X'].unique()
    y_coords = data['Y'].unique()

    # Initialize the result array
    peak_map = np.zeros((len(y_coords), len(x_coords)))

    # Iterate over each point
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            subset = data[(data['X'] == x) & (data['Y'] == y)]
            if subset.empty:
                continue

            wave = subset['Wave'].values
            intensity = subset['Intensity'].values
            intensity = (intensity / np.max(intensity)) * 1000

            try:
                popt, pcov = curve_fit(fitfuncname, wave, intensity)
                perr = np.sqrt(np.diag(pcov))
                if perr[0] / popt[0] < error_tolerance:
                    peak_map[j, i] = popt[0]
                else:
                    peak_map[j, i] = np.nan
            except RuntimeError:
                peak_map[j, i] = np.nan

    # Sift out special points that exceed the threshold and replace them with NaN
    peak_map[(peak_map > threshold_peakmax) | (peak_map < threshold_peakmin)] = np.nan
    plt.imshow(peak_map, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], origin='lower',
               aspect='auto', cmap='plasma')
    plt.colorbar()
    plt.xlabel(Xlable)
    plt.ylabel(Ylable)
    plt.title(Title1)
    plt.show()
    return 0


def drawonepeakwidth(filename,Xlable,Ylable,Title1,threshold_fwhm,fitfuncname,error_tolerance):
    # read data
    data = pd.read_csv(filename, delimiter=r'\s+|\t', engine='python', header=None,
                       names=['X', 'Y', 'Wave', 'Intensity'], skiprows=1)
    # Get unique x and y coordinates
    x_coords = data['X'].unique()
    y_coords = data['Y'].unique()

    # Initialize the result array
    fwhm_map = np.zeros((len(y_coords), len(x_coords)))

    # Iterate over each point
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            subset = data[(data['X'] == x) & (data['Y'] == y)]
            if subset.empty:
                continue

            wave = subset['Wave'].values
            intensity = subset['Intensity'].values
            intensity = (intensity / np.max(intensity)) * 1000

            try:
                popt, pcov = curve_fit(fitfuncname, wave, intensity)
                perr = np.sqrt(np.diag(pcov))
                match Itrans(fitfuncname):
                    case 'voigt':
                        # 这里是筛选不合格数据的标准
                        if perr[0] / popt[0] < error_tolerance:
                            fwhm_map[j, i] = 0.5346 * 2 * popt[3] + np.sqrt(
                                0.2166 * (2 * popt[3]) ** 2 + (2 * popt[2] * np.sqrt(2 * np.log(2))) ** 2)
                        else:
                            fwhm_map[j, i] = np.nan
                    case 'lorentz':
                        if perr[0] / popt[0] < error_tolerance:
                            fwhm_map[j, i] = popt[1]
                        else:
                            fwhm_map[j, i] = np.nan
                    case 'gaussian':
                        if perr[0] / popt[0] < error_tolerance:
                            fwhm_map[j, i] = popt[2]*2.3548
                        else:
                            fwhm_map[j, i] = np.nan
            except RuntimeError:
                fwhm_map[j, i] = np.nan

    # Sift out special points that exceed the threshold and replace them with NaN
    fwhm_map[(fwhm_map > threshold_fwhm) | (fwhm_map < 0)] = np.nan
    plt.imshow(fwhm_map, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], origin='lower',
               aspect='auto', cmap='plasma')
    plt.colorbar()
    plt.xlabel(Xlable)
    plt.ylabel(Ylable)
    plt.title(Title1)
    plt.show()
    return 0
def drawtwopeakposition(filename, Xlable, Ylable, Title1, Title2, threshold_peakmax, threshold_peakmin,
                                       fitfuncname, error_tolerance):
    data = pd.read_csv(filename, delimiter=r'\s+|\t', engine='python', header=None,
                       names=['X', 'Y', 'Wave', 'Intensity'], skiprows=1)
    # Get unique x and y coordinates
    x_coords = data['X'].unique()
    y_coords = data['Y'].unique()

    # Initialize the result array
    peak_map1 = np.zeros((len(y_coords), len(x_coords)))
    peak_map2 = np.zeros((len(y_coords), len(x_coords)))
    # Iterate over each point
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            subset = data[(data['X'] == x) & (data['Y'] == y)]
            if subset.empty:
                continue

            wave = subset['Wave'].values
            intensity = subset['Intensity'].values
            intensity = (intensity / np.max(intensity)) * 1000

            try:
                popt, pcov = curve_fit(fitfuncname, wave, intensity)
                perr = np.sqrt(np.diag(pcov))
                if perr[0] / popt[0] < error_tolerance:
                    peak_map1[j, i] = min(popt[0], popt[1])
                    peak_map2[j, i] = max(popt[0], popt[1])
                else:
                    peak_map1[j, i] = np.nan
                    peak_map2[j, i] = np.nan
            except RuntimeError:
                peak_map1[j, i] = np.nan
                peak_map2[j, i] = np.nan

    # Sift out special points that exceed the threshold and replace them with NaN
    peak_map1[(peak_map1 > threshold_peakmax) | (peak_map1 < threshold_peakmin)] = np.nan
    peak_map2[(peak_map2 > threshold_peakmax) | (peak_map2 < threshold_peakmin)] = np.nan
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    c1 = ax[0].imshow(peak_map1, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()), origin='lower',
                      aspect='auto', cmap='plasma')
    ax[0].set_title(Title1)
    ax[0].set_xlabel(Xlable)
    ax[0].set_ylabel(Ylable)
    fig.colorbar(c1, ax=ax[0])

    c2 = ax[1].imshow(peak_map2, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()), origin='lower',
                      aspect='auto', cmap='plasma')
    ax[1].set_title(Title2)
    ax[1].set_xlabel(Xlable)
    ax[1].set_ylabel(Ylable)
    fig.colorbar(c2, ax=ax[1])

    plt.show()
    return 0
def drawtwopeakwidth(filename,Xlable,Ylable,Title1,Title2, threshold_fwhm, fitfuncname, error_tolerance):
    data = pd.read_csv(filename, delimiter=r'\s+|\t', engine='python', header=None,
                       names=['X', 'Y', 'Wave', 'Intensity'], skiprows=1)
    # Get unique x and y coordinates
    x_coords = data['X'].unique()
    y_coords = data['Y'].unique()

    # Initialize the result array
    fwhm_map1 = np.zeros((len(y_coords), len(x_coords)))
    fwhm_map2 = np.zeros((len(y_coords), len(x_coords)))
    # Iterate over each point
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            subset = data[(data['X'] == x) & (data['Y'] == y)]
            if subset.empty:
                continue

            wave = subset['Wave'].values
            intensity = subset['Intensity'].values
            intensity = (intensity / np.max(intensity)) * 1000

            try:
                popt, pcov = curve_fit(fitfuncname, wave, intensity)
                perr = np.sqrt(np.diag(pcov))
                if perr[0] / popt[0] < error_tolerance:
                    match Itrans(fitfuncname):
                        case 'double_lorentz':
                            fwhm_map1[j, i] = popt[2] if popt[0] < popt[1] else popt[4]
                            fwhm_map2[j, i] = popt[4] if popt[0] < popt[1] else popt[2]
                        case 'double_gaussian':
                            fwhm_map1[j, i] = popt[3]*2.3548 if popt[0] < popt[1] else popt[5]
                            fwhm_map2[j, i] = popt[5]*2.3548 if popt[0] < popt[1] else popt[3]
                        case 'double_voigt':
                            fwhm_map1[j, i] = 0.5346 * 2 * popt[4] + np.sqrt(
                                0.2166 * (2 * popt[4]) ** 2 + (2 * popt[3] * np.sqrt(2 * np.log(2))) ** 2) if popt[0] < popt[1] else 0.5346 * 2 * popt[7] + np.sqrt(
                                0.2166 * (2 * popt[7]) ** 2 + (2 * popt[6] * np.sqrt(2 * np.log(2))) ** 2)
                            fwhm_map2[j, i] = 0.5346 * 2 * popt[4] + np.sqrt(
                                0.2166 * (2 * popt[4]) ** 2 + (2 * popt[3] * np.sqrt(2 * np.log(2))) ** 2) if popt[1] < popt[0] else 0.5346 * 2 * popt[7] + np.sqrt(
                                0.2166 * (2 * popt[7]) ** 2 + (2 * popt[6] * np.sqrt(2 * np.log(2))) ** 2)
                        case 'gaussianpluslorentz':
                            fwhm_map1[j, i] = popt[3]*2.3548 if popt[0]<popt[1] else popt[4]
                            fwhm_map2[j, i] = popt[4] if popt[0]<popt[1] else popt[3]*2.3548
                else:
                    fwhm_map1[j, i] = np.nan
                    fwhm_map2[j, i] = np.nan
            except RuntimeError:
                fwhm_map1[j, i] = np.nan
                fwhm_map2[j, i] = np.nan

    # Sift out special points that exceed the threshold and replace them with NaN
    fwhm_map1[(fwhm_map1 > threshold_fwhm) | (fwhm_map1 < 0)] = np.nan
    fwhm_map2[(fwhm_map2 > threshold_fwhm) | (fwhm_map2 < 0)] = np.nan

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    c1 = ax[0].imshow(fwhm_map1, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
                      origin='lower',
                      aspect='auto', cmap='plasma')
    ax[0].set_title(Title1)
    ax[0].set_xlabel(Xlable)
    ax[0].set_ylabel(Ylable)
    fig.colorbar(c1, ax=ax[0])

    c2 = ax[1].imshow(fwhm_map2, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
                      origin='lower',
                      aspect='auto', cmap='plasma')
    ax[1].set_title(Title2)
    ax[1].set_xlabel(Xlable)
    ax[1].set_ylabel(Ylable)
    fig.colorbar(c2, ax=ax[1])

    plt.show()
    return 0
def drawdelta(filename,Xlable,Ylable,Title1,Title2,threshold_fwhm,threshold_peakmin,threshold_peakmax,fitfuncname,error_tolerance):
    data = pd.read_csv(filename, delimiter=r'\s+|\t', engine='python', header=None,
                       names=['X', 'Y', 'Wave', 'Intensity'], skiprows=1)
    # Get unique x and y coordinates
    x_coords = data['X'].unique()
    y_coords = data['Y'].unique()

    # Initialize the result array
    delta_map = np.zeros((len(y_coords), len(x_coords)))
    ratio_map = np.zeros((len(y_coords), len(x_coords)))
    # Iterate over each point
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            subset = data[(data['X'] == x) & (data['Y'] == y)]
            if subset.empty:
                continue

            wave = subset['Wave'].values
            intensity = subset['Intensity'].values
            intensity = (intensity / np.max(intensity)) * 1000

            try:
                popt, pcov = curve_fit(fitfuncname, wave, intensity)
                perr = np.sqrt(np.diag(pcov))
                p = DCintensity(popt,wave,fitfuncname)
                if perr[0] / popt[0] < error_tolerance:
                    peak_map1 = popt[0]
                    peak_map2 = popt[1]
                    match Itrans(fitfuncname):
                        case 'double_lorentz':
                            fwhm_map1 = popt[2] if popt[0] < popt[1] else popt[4]
                            fwhm_map2 = popt[4] if popt[0] < popt[1] else popt[2]
                        case 'double_gaussian':
                            fwhm_map1 = popt[3] * 2.3548 if popt[0] < popt[1] else popt[5]
                            fwhm_map2 = popt[5] * 2.3548 if popt[0] < popt[1] else popt[3]
                        case 'double_voigt':
                            fwhm_map1 = 0.5346 * 2 * popt[4] + np.sqrt(
                                0.2166 * (2 * popt[4]) ** 2 + (2 * popt[3] * np.sqrt(2 * np.log(2))) ** 2)
                            fwhm_map2 = 0.5346 * 2 * popt[7] + np.sqrt(
                                0.2166 * (2 * popt[7]) ** 2 + (2 * popt[6] * np.sqrt(2 * np.log(2))) ** 2)
                        case 'gaussianpluslorentz':
                            fwhm_map1 = popt[3] * 2.3548 if popt[0] < popt[1] else popt[4]
                            fwhm_map2 = popt[4] if popt[0] < popt[1] else popt[3] * 2.3548
                    delta_map[j, i] = np.abs(popt[1]-popt[0])
                    ratio_map[j, i] = p[0]/p[1]
                    if (peak_map1 > threshold_peakmax) | (peak_map1 < threshold_peakmin):
                        delta_map[j, i] = np.nan
                        ratio_map[j, i] = np.nan
                    if (peak_map2 > threshold_peakmax) | (peak_map2 < threshold_peakmin):
                        delta_map[j, i] = np.nan
                        ratio_map[j, i] = np.nan
                    if (fwhm_map1 > threshold_fwhm) | (fwhm_map1 < 0):
                        delta_map[j, i] = np.nan
                        ratio_map[j, i] = np.nan
                    if (fwhm_map2 > threshold_fwhm) | (fwhm_map2 < 0):
                        delta_map[j, i] = np.nan
                        ratio_map[j, i] = np.nan
                else:
                    delta_map[j, i] = np.nan
                    ratio_map[j, i] = np.nan
            except RuntimeError:
                delta_map[j, i] = np.nan
                ratio_map[j, i] = np.nan


    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    c1 = ax[0].imshow(delta_map, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
                      origin='lower',
                      aspect='auto', cmap='plasma')
    ax[0].set_title(Title1)
    ax[0].set_xlabel(Xlable)
    ax[0].set_ylabel(Ylable)
    fig.colorbar(c1, ax=ax[0])

    c2 = ax[1].imshow(ratio_map, extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
                      origin='lower',
                      aspect='auto', cmap='plasma')
    ax[1].set_title(Title2)
    ax[1].set_xlabel(Xlable)
    ax[1].set_ylabel(Ylable)
    fig.colorbar(c2, ax=ax[1])

    plt.show()
    return 0
def DCintensity(para,wave,fitfuncname):
    match Itrans(fitfuncname):
        case 'double_lorentz':
            y1 = lorentz(wave, para[0],para[2],para[3]) if para[0]<para[1] else lorentz(wave, para[1], para[4],para[5])
            y2 = lorentz(wave, para[1], para[4],para[5]) if para[0]<para[1] else lorentz(wave, para[0],para[2],para[3])
        case 'double_gaussian':
            y1 = gaussian(wave, para[0],para[2],para[3]) if para[0]<para[1] else gaussian(wave, para[1], para[4],para[5])
            y2 = gaussian(wave, para[1], para[4],para[5]) if para[0]<para[1] else gaussian(wave, para[0],para[2],para[3])

        case 'double_voigt':
            y1 = voigt(wave,para[0],para[2],para[3],para[4]) if para[0]<para[1] else voigt(wave,para[1],para[5],para[6],para[7])
            y2 = voigt(wave, para[0], para[2], para[3], para[4]) if para[1] < para[0] else voigt(wave, para[1], para[5],para[6], para[7])
        case 'gaussianpluslorentz':
            y1 = gaussian(wave,para[0],para[2],para[3]) if para[0]<para[1] else lorentz(wave,para[1],para[4],para[5])
            y2 = gaussian(wave,para[0],para[2],para[3]) if para[1]<para[0] else lorentz(wave,para[1],para[4],para[5])

    A = [np.trapz(y1, wave),np.trapz(y2,wave)]
    return A
import tkinter as tk
from tkinter import filedialog


def submit():
    file_path = file_path_var.get()
    peak_number = peak_number_var.get()
    function = function_var.get()
    title1 = title1_var.get()
    title2 = title2_var.get()
    threshold_peak_pos_max = threshold_peak_pos_max_var.get()
    threshold_peak_pos_min = threshold_peak_pos_min_var.get()
    xlabel = xlabel_var.get()
    ylabel = ylabel_var.get()
    threshold_peak_width = threshold_peak_width_var.get()

    filename = file_path_var.get()
    peak_num = int(peak_number_var.get())
    You_Draw = You_draw_var.get()
    Xlable = xlabel_var.get()
    Ylable = ylabel_var.get()
    Title1 = title1_var.get()
    Title2 = title2_var.get()
    try:
        threshold_fwhm = float(threshold_peak_width_var.get())
    except ValueError:
        threshold_fwhm = 0

    try:
        threshold_peakmax = float(threshold_peak_pos_max_var.get())
    except ValueError:
        threshold_peakmax = 0

    try:
        threshold_peakmin = float(threshold_peak_pos_min_var.get())
    except ValueError:
        threshold_peakmin = 0

    fitfuncname = trans(function_var.get())

    try:
        error_tolerance = float(error_tolerance_var.get())
    except ValueError:
        error_tolerance = 1

    if peak_num == 1:
        match You_Draw:
            case 'intensity':
                drawonepeakint(filename, Xlable, Ylable, Title1)
            case 'position':
                drawonepeakposition(filename, Xlable, Ylable, Title1, threshold_peakmax, threshold_peakmin,
                                       fitfuncname, error_tolerance)
            case 'width':
                drawonepeakwidth(filename, Xlable, Ylable, Title1, threshold_fwhm, fitfuncname, error_tolerance)
            case _:
                print("Wrong!")
    else:
        if peak_num == 2:
            match You_Draw:
                case 'twopeakpositon':
                    drawtwopeakposition(filename,Xlable,Ylable,Title1,Title2,threshold_peakmax, threshold_peakmin, fitfuncname, error_tolerance)
                case 'twopeakwidth':
                    drawtwopeakwidth(filename,Xlable,Ylable,Title1,Title2,threshold_fwhm,fitfuncname,error_tolerance)
                case 'position difference and intensity ratio':
                    drawdelta(filename,Xlable,Ylable,Title1,Title2,threshold_fwhm,threshold_peakmin,threshold_peakmax,fitfuncname,error_tolerance)
        else:
            print("not a available value")


app = tk.Tk()
app.title("Heat Map")

# File path selection
file_path_var = tk.StringVar()
tk.Label(app, text="File Path:").grid(row=0, column=0)
tk.Entry(app, textvariable=file_path_var).grid(row=0, column=1)
tk.Button(app, text="Browse", command=lambda: file_path_var.set(filedialog.askopenfilename())).grid(row=0, column=2)

# Peak number selection
peak_number_var = tk.StringVar(value="1")
tk.Label(app, text="Peak Number:").grid(row=1, column=0)
tk.OptionMenu(app, peak_number_var, "1", "2").grid(row=1, column=1)

# Function selection
function_var = tk.StringVar(value="lorentz")
tk.Label(app, text="Function:").grid(row=2, column=0)
tk.OptionMenu(app, function_var, 'lorentz', 'gaussian', 'voigt', 'double_lorentz', 'double_gaussian', 'double_voigt', 'gaussianpluslorentz').grid(row=2, column=1)

# You draw
You_draw_var = tk.StringVar(value='intensity')
tk.Label(app, text="You draw:").grid(row=3,column=0)
tk.OptionMenu(app, You_draw_var, 'intensity','position', 'width','twopeakpositon','twopeakwidth',"position difference and intensity ratio").grid(row=3,column=1)
# Threshold of peak position - max
threshold_peak_pos_max_var = tk.StringVar()
tk.Label(app, text="Threshold of Peak Position - Max:").grid(row=4, column=0)
tk.Entry(app, textvariable=threshold_peak_pos_max_var).grid(row=4, column=1)

# Threshold of peak position - min
threshold_peak_pos_min_var = tk.StringVar()
tk.Label(app, text="Threshold of Peak Position - Min:").grid(row=5, column=0)
tk.Entry(app, textvariable=threshold_peak_pos_min_var).grid(row=5, column=1)

# Threshold of peak width
threshold_peak_width_var = tk.StringVar()
tk.Label(app, text="Threshold of Peak Width:").grid(row=6, column=0)
tk.Entry(app, textvariable=threshold_peak_width_var).grid(row=6, column=1)

#
error_tolerance_var =  tk.StringVar()
tk.Label(app, text="Error Tolerance:").grid(row=7,column=0)
tk.Entry(app, textvariable=error_tolerance_var).grid(row=7,column=1)
# Title1 input
title1_var = tk.StringVar()
tk.Label(app, text="Title1:").grid(row=8, column=0)
tk.Entry(app, textvariable=title1_var).grid(row=8, column=1)

# Title2 input
title2_var = tk.StringVar()
tk.Label(app, text="Title2(optional):").grid(row=9, column=0)
tk.Entry(app, textvariable=title2_var).grid(row=9, column=1)

# X Label
xlabel_var = tk.StringVar()
tk.Label(app, text="X Label:").grid(row=10, column=0)
tk.Entry(app, textvariable=xlabel_var).grid(row=10, column=1)

# Y Label
ylabel_var = tk.StringVar()
tk.Label(app, text="Y Label:").grid(row=11, column=0)
tk.Entry(app, textvariable=ylabel_var).grid(row=11, column=1)

# Submit button
tk.Button(app, text="Submit", command=submit).grid(row=12, column=1)

app.mainloop()