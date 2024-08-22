# import matplotlib.pyplot as plt
# from astropy.table import Table

# tab = Table.read(r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\info.dat', format='ascii')

# mask = (tab['flares_present'] == 'True')

# x=tab['t_decay'][mask]

# y=tab['peak_flux_ref'][mask]

# plt.hist2d(x,y)

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# Load the table
tab = Table.read(r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\info.dat', format='ascii')

# Convert relevant columns to numpy arrays
t_decay = np.array(tab['t_decay'])
peak_flux_ref = np.array(tab['peak_flux_ref'])
flares_present = np.array(tab['flares_present'])

# Define the number of bins for the histogram
bins = 50  # Adjust as needed

# 2D histogram for all IDs
hist_total, xedges, yedges = np.histogram2d(t_decay, peak_flux_ref, bins=bins)

# 2D histogram for IDs with flares_present == True
mask = (flares_present == 'True')
hist_flares, _, _ = np.histogram2d(t_decay[mask], peak_flux_ref[mask], bins=[xedges, yedges])

# Calculate the fraction
fraction = np.divide(hist_flares, hist_total, out=np.zeros_like(hist_flares, dtype=float), where=hist_total != 0)

# Plotting the fraction as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(fraction.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='viridis')
plt.colorbar(label='Fraction of flares_present == True')
plt.xlabel('t_decay')
plt.ylabel('Peak Flux in reference band')
plt.title('Fraction of Flares Present (True) in 2D Histogram')
plt.show()
