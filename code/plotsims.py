import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import re

# Define file paths
file_paths = [
    r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\1.dat'
    
    # T0, Sigma Rise constant. Varying t_peak, peak flux and t_decay. (Files 4-7)
    # r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_9_2_con1.dat',
    # r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_9_4_con1.dat',
    #  r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_20_2_con1.dat',
    #  r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_50_2_con1.dat',

    # T0, Sigma Rise, T_peak constant. Varying peak flux and t_decay. (Files 8-11)
    # r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_9_2_con2.dat'

    # All Randoms with different filters (Files 0-3)
    # r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_9_2.dat',
    # r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_20_2.dat',
    # r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\simulations\r1+d1_50_2.dat'
]


# Number of files
num_files = len(file_paths)

# Determine the number of rows needed for the subplots
rows = num_files
cols = 2  # Two columns: one for total number of IDs, one for fraction

# Create a figure with a dynamic number of subplots based on the number of files
fig, ax = plt.subplots(rows, cols, figsize=(12, 6 * rows), sharex=True, sharey=True)

# Handle the case where there's only one file (and thus, one row of subplots)
if num_files == 1:
    ax = np.array([ax])  # Convert to a 2D array for consistent indexing

bins = 10  # Number of bins for histograms

for idx, file_path in enumerate(file_paths):
    # Load the table
    tab = Table.read(file_path, format='ascii')
    
    # Extract T and alpha from filename using regex
    match = re.search(r'r1\+d1_(\d+)_(\d+)(?:_con\d+)?\.dat', file_path)
    if match:
        T = match.group(1)
        alpha = int(match.group(2)) / 10  # Divide Y by 10 to get alpha
    else:
        T = 'Unknown'
        alpha = 'Unknown'
    
    filtered_tab = tab

    # Convert relevant columns to numpy arrays
    t_decay = np.array(filtered_tab['t_decay'])
    peak_flux_ref = np.array(filtered_tab['peak_flux_ref'])
    flares_present = np.array(filtered_tab['flares_present'])

    # Calculate 2D histograms
    hist_total, xedges, yedges = np.histogram2d(t_decay, peak_flux_ref, bins=bins)
    mask = (flares_present == 'True')
    hist_flares, _, _ = np.histogram2d(t_decay[mask], peak_flux_ref[mask], bins=[xedges, yedges])
    fraction = np.divide(hist_flares, hist_total, out=np.zeros_like(hist_flares, dtype=float), where=hist_total != 0)

    # Plot the total number of IDs
    im1 = ax[idx, 0].imshow(hist_total.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='viridis')
    ax[idx, 0].set_title(f'Number of simulations')
    ax[idx, 0].set_xlabel('peak flare epoch')
    ax[idx, 0].set_ylabel('peak flux in reference band (uJy)')
    fig.colorbar(im1, ax=ax[idx, 0], label='Number of simulations')

    # Plot the fraction of flares present
    im2 = ax[idx, 1].imshow(fraction.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='viridis')
    ax[idx, 1].set_title(f'Fraction of simulations with flares detected')
    ax[idx, 1].set_xlabel('peak flare epoch')
    fig.colorbar(im2, ax=ax[idx, 1], label='Fraction of simulations with flares detected')

# Add a global title and adjust layout
#plt.suptitle('t_decay vs. peak_flux_ref', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
