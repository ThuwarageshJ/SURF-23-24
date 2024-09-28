# %reload_ext autoreload
# %autoreload 2

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
import time
# Columns and corresponding datatypes for lightcurves.
cols = 'jd,mag,magerr,filt,field,flux,fluxerr,adjflux'
dtypes = 'int,float,float,str,int,float,float,float'
t0=time.time()
# Unique IDs of data
#ids = np.unique([f.split('\\')[-1].split('_')[0] for f in glob.glob(f'{cur_folder_path}/forced_lc/*.dat')])[5:]
ids=['108602273971326964','108592173310873531','82973163747267487', '94300438321684163']
# ids=[ '94300438321684163']
# trials=np.random.randint(0, 100, size=100)

for id in tqdm(ids):
    print(id)
    # if glob.glob(f'{cur_folder_path}/pickles/{id}.pickle'):
    #     print("\nRetrieving pickle.. ", id)
    #     filepath=os.path.join(f'{cur_folder_path}/pickles', id+'.pickle')
    #     with open(filepath, 'rb') as file:
    #         LC=pickle.load(file)
    #     LC.find_flare()
    #     LC.plot(show= True, save=False, save_loc=f'{cur_folder_path}/samples')
    #     continue 
    
    lc = None

    # if glob.glob(f'{cur_folder_path}/forced_lc_by_id/{id}.dat'): 
    if False:

        # Use the processed light curve file, if it already exists
        lc = Table.read(f'{cur_folder_path}/forced_lc_by_id/{id}.dat', format='ascii')

    else:

        # Create lightcurve table with data from all files of the current ID
        for f in glob.glob(f'{cur_folder_path}/forced_lc/{id}*.dat'):

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

        # #  # Sort by filter and then by jd
        # lc.sort(['filt', 'adjflux'])
        # # print(lc)
        
        # for filt in ['zg', 'zr']:
        #     #print(filt)
        #     mask = (lc['filt']==filt)
        #     #print(lc['adjflux'][mask][:N_low])
        #     zp_median = np.median(lc['adjflux'][mask][:N_low])
        #     #print(zp_median)
        #     lc['adjflux'][mask]=lc['adjflux'][mask]-zp_median
        
        lc.sort(['filt', 'jd'])

        # Save the processed light curve data
        if lc!=None:
            ascii.write(lc, f'{cur_folder_path}/forced_lc_by_id/{id}.dat', overwrite=True)
    
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
            # LC.plot(show=1, save=0, plot_std = 0,plot_data=1, save_loc=f'{cur_folder_path}/samples')
            # LC.plot(show=1, save=0, plot_std = 1,plot_data=0, save_loc=f'{cur_folder_path}/samples')
            LC.find_flare(user=False, print_params=print_flare_parameters)
            LC.plot(show=1, save=0, plot_std = 0,plot_data=1, save_loc=f'{cur_folder_path}/samples')
            LC.plot(show=1, save=0, plot_std = 1,plot_data=0, save_loc=f'{cur_folder_path}/samples')
            

            # print("\n Saving the lightcurve object as %s..." % id)
            # filepath=os.path.join(f'{cur_folder_path}/pickles', id+'.pickle')
            # with open(filepath, 'wb') as file:
            #     pickle.dump(LC, file)

print(time.time()-t0)