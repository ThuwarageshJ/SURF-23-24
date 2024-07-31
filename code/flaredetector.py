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

def light_curve(id):
    
    # Uncomment if using intermediate files

    # if glob.glob(f'{cur_folder_path}/pickles/{id}.pickle'):
    #     print("\nRetrieving pickle.. ", id)
    #     filepath=os.path.join(f'{cur_folder_path}/pickles', id+'.pickle')
    #     with open(filepath, 'rb') as file:
    #         LC=pickle.load(file)
    #     LC.find_flare()
    #     LC.plot(show= True, save=False, save_loc=f'{cur_folder_path}/samples')
    #     continue 
    
    lc = None

    # Uncomment if using intermediate files
    # if glob.glob(f'{temp_path}/{id}.dat'): 

    #     # Use the processed light curve file, if it already exists
    #     lc = Table.read(f'{temp_path}/{id}.dat', format='ascii')

    # else:

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

    # Uncomment to store intermediate files
    # Save the processed light curve data
    # if lc!=None:
    #     ascii.write(lc, f'{temp_path}/{id}.dat', overwrite=True)

    return lc

def process_light_curve(lc, adjust_parameters, reset_params, show, save, plot_std, fig_path, pickle_path):

    if lc!=None:

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

            # Change lines below to make adjustments.
            LC.find_flare(user=adjust_parameters, reset_params=reset_params, print_params=print_flare_parameters)                                    # user = True enables user to manually change T and alpha values               
            if adjust_parameters and reset_params:
                LC.find_flare(user=False)
            LC.plot(show=show, save=save, plot_std = plot_std, save_loc=fig_path)    # See descriptions in lightcurve class plot() method
            if save_pickle and pickle_path is not None:
                LC.save_pickle(pickle_path=pickle_path)                                 # Remove pickle_path parameter to avoid saving pickle files

def divide_files_into_batches(file_list, batch_size):

    """Divide the file list into batches of a given size."""
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]

def process_batch(batch):
    """Process a batch of files."""
    for id in tqdm(batch):
        # Replace this with actual file processing logic
        print(f"Processing {id}")
        print(type(id))
        lc = light_curve(id)
        process_light_curve(lc, adjust_parameters, reset_params, show, save, plot_std, fig_path, pickle_path)


# Unique IDs of data
ids = np.unique([f.split('\\')[-1].split('_')[0] for f in glob.glob(f'{lc_path}/*.dat')])[:100]

# file_list = [f"file_{i}.txt" for i in range(500000)]  # List of 500,000 file names
# Adjust batch size based on your system's memory and processing capability
batches = list(divide_files_into_batches(ids, batch_size))

# Uncomment the following lines to process only specified files OR a random set of files
# ids=['108602273971326964','108592173310873531','82973163747267487', '94300438321684163']
# random=np.random.randint(0, len(ids), size=100)

def main():

    num_batches = len(batches)
    num_workers = min(4, num_batches)  # Number of workers should be chosen based on your system

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
    main()
    print(time.time()-t0)



    