import tkinter as tk
from tkinter import filedialog
import function as fc


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

    fitfuncname = fc.trans(function_var.get())

    try:
        error_tolerance = float(error_tolerance_var.get())
    except ValueError:
        error_tolerance = 1

    if peak_num == 1:
        match You_Draw:
            case 'intensity':
                fc.drawonepeakint(filename, Xlable, Ylable, Title1)
            case 'position':
                fc.drawonepeakposition(filename, Xlable, Ylable, Title1, threshold_peakmax, threshold_peakmin,
                                       fitfuncname, error_tolerance)
            case 'width':
                fc.drawonepeakwidth(filename, Xlable, Ylable, Title1, threshold_fwhm, fitfuncname, error_tolerance)
            case _:
                print("Wrong!")
    else:
        if peak_num == 2:
            match You_Draw:
                case 'twopeakpositon':
                    fc.drawtwopeakposition(filename,Xlable,Ylable,Title1,Title2,threshold_peakmax, threshold_peakmin, fitfuncname, error_tolerance)
                case 'twopeakwidth':
                    fc.drawtwopeakwidth(filename,Xlable,Ylable,Title1,Title2,threshold_fwhm,fitfuncname,error_tolerance)
                case 'position difference and intensity ratio':
                    fc.drawdelta(filename,Xlable,Ylable,Title1,Title2,threshold_fwhm,threshold_peakmin,threshold_peakmax,fitfuncname,error_tolerance)
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

