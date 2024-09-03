# SURF-23-24
Code for SURF Project 2023-24

# Instructions to run flaredetector.py
- Make sure `flaredetector.py `, `lightcurveprocessor.py`, `constants.py` are in the same location/folder.
- Changes need to be done only in `constants.py` (Mostly will need to change only lines 8-19).
- Lines 8-10:
    - Specify no. of cores, no. of batches, and whether to proceed with multi processing or not.
- Lines 12-15:
    - Enter folder paths to retrieve light curves from, save plots, and save pickle files. Ignore temp_path
- Lines 17-19:
    - Specify whether to show plots, save plots, and save pickle files
- Lines 21-25 (Optional):
    - Customize flare detection and plots
- Lines 30-35 (Optional):
    - Customize parameters for flare detection and light curve processing
- Run `constants.py` after mentioning the folderpaths to check the validity of the paths.

# Instructions to create simulations (simulator5.py)
- Specify the no. of simulations to be created from a light curve file (line 48) and the folder from which to draw raw files for simulations (line 13) in `constants.py`. (For example raw files, see code/forced_lc).
- Run `simulator5.py`. This will implement the following steps.
    1. Parse data from raw files with same ID; Combine data from all the filters and fields; Implement zero point corrections.
    2. Flatten out the light curve by replacing the flux values with values drawn from a normal distribution of mean 0.
    3. Simulate r1+d1 flares of various randomized parameters sets and add to the flat curves.
    4. Run the flare detector on the raw (flat) light curve and all the simulations obtained from it.
    5. Save the data files, plots, and pickle files of simulations, and the plots of the raw files in `code\simulations`. 
    6. The information about all the simulations generated is temporarily stored as a table in `code\simulations\info.dat`.
    7. After the program has been exited, this data is appended to `code\simulations\info_perm.dat`. (This is to have a permanent storage space if the code is run batch by batch)
- Run `plotsims.py` to see the 2d scatter plots.