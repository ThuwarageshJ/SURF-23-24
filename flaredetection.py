# %reload_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sys, os, glob
from astropy.table import Table
from astropy.io import ascii
from tqdm import *
from gaussianfittemp import lightcurve

# Columns and corresponding datatypes for lightcurves.
cols = 'jd,mag,magerr,filt,field,flux,fluxerr,adjflux'
dtypes = 'int,float,float,str,int,float,float,float'

# Unique IDs of data
ids = np.unique([f.split('\\')[-1].split('_')[0] for f in glob.glob('C:/Users/thuwa/Coding/SURF/SURF-23-24/forced_lc/*.dat')])

for id in tqdm(ids):

    lc = None

    if glob.glob(f'C:/Users/thuwa/Coding/SURF/SURF-23-24/forced_lc_by_id_1/{id}.dat'): 

        # Use the processed light curve file, if it already exists
        lc = Table.read(f'C:/Users/thuwa/Coding/SURF/SURF-23-24/forced_lc_by_id_1/{id}.dat', format='ascii')

    else:

        # Create lightcurve table with data from all files of the current ID
        for f in glob.glob(f'C:/Users/thuwa/Coding/SURF/SURF-23-24/forced_lc/{id}*.dat'):

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

                        if (jd-2400000.5)<58500:                        # add to zero point calculation if mjd<58500
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
        
        # Sort by filter and then by jd
        lc.sort(['filt', 'jd'])

        # Save the processed light curve data
        if lc!=None:
            ascii.write(lc, f'C:/Users/thuwa/Coding/SURF/SURF-23-24/forced_lc_by_id_1/{id}.dat', overwrite=True)
    
    if lc!=None:

        if len(lc)>0:

            timeseries=dict()
            data=dict()
            dataerr=dict()

            for f in np.unique(lc['filt']):

                mask = (lc['filt']==f)

                timeseries[f]=np.array(lc['jd'][mask]-2400000.5)
                data[f]=np.array(lc['adjflux'][mask])
                dataerr[f]=np.array(lc['fluxerr'][mask])
            
            # print(timeseries)
            # print(data)
            # print(dataerr)
    

            LC=lightcurve(timeseries, data , dataerr, id)
            #LC.regress()
            # #LC.plot()
            # #LC.findflare()
            LC.plot(show=True)
                