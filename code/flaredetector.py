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
from constants import *
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed


for i in range(2):
    if not os.path.exists(pickle_paths[i]) and save_pickle:
        os.makedirs(pickle_paths[i])
    if not os.path.exists(fig_paths[i]) and save:
        os.makedirs(fig_paths[i])

def light_curve(id):
    
    lc = None

    # Create lightcurve table with data from all files of the current ID
    for f in glob.glob(f'{lc_path}/{id}*.dat'):

        filt = f.split('_')[-1].split('.')[0]       # ZTF band
        field = f.split('_')[-2]                    # field

        # Create the table if first iteration
        if lc is None:      
            lc = Table(names=cols.split(','), dtype=dtypes.split(',')) 

        # Add rows to the table from the dat files. Discard rows with NaN
        try:

            tab = Table.read(f, format='ascii')     # read from file

            # Summation variables for zero point calculation
            weighted_sum=0.0    
            weight=0.0

            for jd, mag, magerr in tab[['col1','col2','col3']]:

                if np.sum(np.isnan([mag, magerr]))==0:  # eliminate rows with NaN values

                    flux = 3631e6*10**(-0.4*mag)                    # mag to flux   
                    fluxerr = flux*0.4*np.log(10)*magerr            # magerr to fluxerr

                    lc.add_row([jd, mag, magerr, filt, field, flux, fluxerr, 0.0])  # add row

                    if (jd-mjd_adjustment)<mjd_zp:                        # add to zero point calculation if mjd<58500
                        weighted_sum+=(flux/fluxerr**2)
                        weight+=(1/fluxerr**2)
            
            # Calculate zero point
            if weight==0.0:
                zp=0
            else:
                zp=weighted_sum/weight
            
            # Adjusted flux column using zero point deduction
            mask = (lc['filt'] == filt)*(lc['field']==int(field))
            lc['adjflux'][mask]=lc['flux'][mask]-zp

        except Exception as e:
            continue #print(e)
    
    # Uncomment the following section go adjust zero point based on the median of the lowest N_low values in each filter
    # Sort by filter and then by adjusted flux
    # lc.sort(['filt', 'adjflux'])
    
    # for filt in ['zg', 'zr']:
    #     mask = (lc['filt']==filt)
    #     zp_median = np.median(lc['adjflux'][mask][:N_low])
    #     lc['adjflux'][mask]=lc['adjflux'][mask]-zp_median
    
    lc.sort(['filt', 'jd'])

    return lc

def process_light_curve(id, lc, adjust_parameters, reset_params, show, save, plot_std, fig_paths, pickle_paths):

    if lc is None:
        return
    
    if len(lc)>0:

        timeseries=dict()
        data=dict()
        dataerr=dict()

        for f in np.unique(lc['filt']):

            mask = (lc['filt']==f)

            timeseries[f]=np.array(lc['jd'][mask]-mjd_adjustment)
            data[f]=np.array(lc['adjflux'][mask])
            dataerr[f]=np.array(lc['fluxerr'][mask])
        
        LC=LightCurve(timeseries, data , dataerr, id)

        LC.find_flare(user=adjust_parameters, reset_params=reset_params, print_params=print_flare_parameters)                                    # user = True enables user to manually change T and alpha values               
        if adjust_parameters and reset_params:
            LC.find_flare(user=False)
        i=0 if LC.flares_present else 1
        LC.plot(show=show, save=save, plot_std = plot_std, save_loc=fig_paths[i])    # See descriptions in lightcurve class plot() method
        if save_pickle and pickle_path is not None:
            LC.save_pickle(pickle_path=pickle_paths[i])                                 # Remove pickle_path parameter to avoid saving pickle files

def divide_files_into_batches(file_list, batch_size):

    """Divide the file list into batches of a given size."""
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]

def process_batch(batch):

    """Process a batch of files."""
    for id in tqdm(batch):
        print(f"Processing {id}")
        lc = light_curve(id)
        process_light_curve(id, lc, adjust_parameters, reset_params, show, save, plot_std, fig_paths, pickle_paths)


# Unique IDs of data
ids = np.unique([f.split('\\')[-1].split('_')[0] for f in glob.glob(f'{lc_path}/*.dat')])[:].tolist()

# Uncomment the following lines to process only specified files OR a random set of files
# ids=['108602273971326964','108592173310873531','82973163747267487', '94300438321684163']
# random=np.random.randint(0, len(ids), size=100)

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
        for id in tqdm(ids):
            print(f"Processing {id}")
            lc = light_curve(id)
            process_light_curve(id, lc, adjust_parameters, reset_params, show, save, plot_std, fig_paths, pickle_paths)
    
    print(time.time()-t0)
    



    