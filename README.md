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
- Run `constants'py` after mentioning the folderpaths to check the validity of the paths.
