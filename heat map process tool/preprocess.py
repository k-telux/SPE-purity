import pandas as pd
import numpy as np
import func as f

file1 = 'map_1_pl_1.8eV.txt'
file2 = 'map_2_pl.txt'
file1_min = 1.7
file1_max = 1.87
file2_min = 1.87
file2_max = 2.1
outputname = 'output_data.txt'
# Read two data files
df1 = f.filtering(file1, file1_min, file1_max)
df2 = f.filtering(file2, file2_min, file2_max)


# Splicing two data frames
df_combined = np.concatenate([df1, df2])

# Saving of processed data
np.savetxt(outputname, df_combined, delimiter=' ', fmt='%.4f')
