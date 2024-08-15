import matplotlib.pyplot as plt
import numpy as np
import os

file ='Point.txt' 
data = np.loadtxt(file, delimiter='\t')
x_values = data[:, 2]    
y_values = data[:, 3]
plt.plot(x_values, y_values)
plt.xlabel('Raman Shift (cm-1)')
plt.ylabel('Intensity (a.u.)')
plt.title(f'Data Plot - {file}')
plt.savefig(f'{file.split(".")[0]}.png')
plt.clf()  # Clear the current figure
plt.show()
