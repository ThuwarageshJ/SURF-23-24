# %reload_ext autoreload
# %autoreload 2

import math
import time
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sys, os, glob
from astropy.table import Table
from astropy.io import ascii
from tqdm import *
from lightcurveprocessor import LightCurve
from flaredetector import light_curve, divide_files_into_batches
from constants import *
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

def verify_light_curve_eligibility(lc):
    return True

def simulate(lc):
    pass

def process_batch(batch):

    """Process a batch of files."""
    for id in tqdm(batch):
        print(f"Processing {id}")
        lc = light_curve(id)
        if verify_light_curve_eligibility(lc):
            simulate(lc)

# Unique IDs of data
ids = np.unique([f.split('\\')[-1].split('_')[0] for f in glob.glob(f'{lc_path}/*.dat')]).tolist()

# Divide into batches
batches = list(divide_files_into_batches(ids, batch_size))

def main():

    num_batches = len(batches)
    num_workers = min(cores, num_batches)  # Number of cores

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        
        # Optionally, handle the results or check for exceptions
        for future in as_completed(futures):
            try:
                future.result()  # Block until the result is available
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__=="__main__":

    t0=time.time()

    if use_multiprocessing:
        main()
    else:
        process_batch(ids)
    
    print(time.time()-t0)
    



    