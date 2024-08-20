import numpy as np

def filtering(filename,min,max):

    data = np.genfromtxt(filename, skip_header=1)
    filtered_data = data[(data[:, 2] >= min) & (data[:, 2] <= max)]
    return filtered_data