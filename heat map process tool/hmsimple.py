import numpy as np
import matplotlib.pyplot as plt

filename = 'output_data.txt'
Xlable = 'X Coordinate'
Ylable = 'Y Coordinate'
Title = 'Heat Map of Integrated Spectral Intensity'
colorbarname = 'Integrated Intensity'




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
plt.imshow(heatmap, extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()], origin='lower', aspect='auto',cmap='plasma')
plt.colorbar(label=colorbarname)
plt.xlabel(Xlable)
plt.ylabel(Ylable)
plt.title(Title)
plt.show()
