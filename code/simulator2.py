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
from flaredetector import light_curve, divide_files_into_batches, process_light_curve
from model import model
from constants import *
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

def verify_light_curve_eligibility(lc):

    valid = True

    for f in np.unique(lc['filt']):
        mask = (lc['filt']==f)
        stdev = np.std(lc['adjflux'][mask])
        mean  = np.mean(lc['adjflux'][mask])
        if np.max(lc['adjflux'][mask])>mean+5*stdev or np.min(lc['adjflux'][mask])<mean-5*stdev or np.max(lc['adjflux'][mask])>50 or np.min(lc['adjflux'][mask])<-50:
            valid = False
            break

    return valid

def simulate(lc):

    timeseries=dict()
    data=dict()
    dataerr=dict()

    t_start = np.inf
    t_end = -np.inf

    for f in np.unique(lc['filt']):

        mask = (lc['filt']==f)

        timeseries[f]=np.array(lc['jd'][mask]-mjd_adjustment)
        data[f]=np.array(lc['adjflux'][mask])
        dataerr[f]=np.array(lc['fluxerr'][mask])

        t_start=min(t_start, timeseries[f][0])
        t_end=max(t_end, timeseries[f][-1])
 
    parameters = np.random.randint([0, peak_flux_ref_range[0], T0_range[0], time_range[0], time_range[0]],
                                    [t_end-t_start+1, peak_flux_ref_range[1], T0_range[1], time_range[1], time_range[1]],
                                    size=(n_sims_per_sample,5) )
    
    sims=[]

    for para_set in parameters:


        simulated_data = model(data, timeseries, 'r1', 'd1', 
                                para_set[0], para_set[1], para_set[2], sigma_rise= para_set[3], t_decay = para_set[4])

        para_set=para_set.astype('float64')

        para_set[0]=float(para_set[0]/(t_end-t_start))

        sim = Table(names=cols1.split(','), dtype=dtypes1.split(',')) 

        for f in simulated_data.keys():
            for (i, t) in enumerate(timeseries[f]):
                sim.add_row([t+mjd_adjustment, f, dataerr[f][i], simulated_data[f][i]])
        
        sims.append((sim, para_set))


    return sims


  
def process_batch(paths):

    """Process a batch of files."""

    simulations_info = Table(names=('id', 't_peak', 'peak_flux_ref', 'T0', 'sigma_rise', 't_decay', 'flares_present'), 
                             dtype=('str','float','int','int','int', 'int', 'bool'))
    param_info=Table.read(f'{simulations_path}/info_perm.dat', format='ascii')

    for path in paths:
        ids = np.unique([f for f in glob.glob(f'{path}/*.pickle')]).tolist()
        for id in tqdm(ids):
            print(f"Processing {id.split('\\')[-1].split('.')[0]}")
            LC=LightCurve.get_pickle(id)
            LC.reset_params()
            LC.flares_present=False
            LC.find_flare()
            #lc = Table.read(f'{simulations_path}/data/{id}.dat', format='ascii')
            # raw= process_light_curve(id, lc, adjust_parameters=False, reset_params=True, show = False, save = True, save_pickle=False, plot_std=False, fig_paths=sim_raw_fig_paths, pickle_paths=sim_pickle_paths)
            #flare= process_light_curve(f'{id}', lc, adjust_parameters=False, reset_params=True, show = False, save = False, save_pickle=False, plot_std=False, fig_paths=sim_fig_paths, pickle_paths=sim_pickle_paths)
            mask=(param_info['id']==id.split('\\')[-1].split('.')[0])
            parameters=param_info[mask]
            if len(parameters)>0:
                simulations_info.add_row([f'{id.split('\\')[-1].split('.')[0]}', parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5], LC.flares_present])
                ascii.write(simulations_info, f'{simulations_path}/info.dat', overwrite=True)

# Unique IDs of data
ids = np.unique([f.split('\\')[-1].split('.')[0] for f in glob.glob(f'{simulations_path}/data/*.dat')]).tolist()


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
        process_batch(sim_pickle_paths)
    
    print(time.time()-t0)
    



    