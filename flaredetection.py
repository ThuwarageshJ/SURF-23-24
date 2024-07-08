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
cols = 'jd,mag,magerr,filt,field'
dtypes = 'int,float,float,str,int'

# Unique IDs of data
ids = np.unique([f.split('\\')[-1].split('_')[0] for f in glob.glob('C:/Users/thuwa/Coding/SURF/forced_lc/*.dat')])

for id in tqdm(ids[3:]):

    lc = None

    if glob.glob(f'C:/Users/thuwa/Coding/SURF/forced_lc_by_id/{id}.dat'): 

        # Use the processed light curve file, if it already exists
        lc = Table.read(f'C:/Users/thuwa/Coding/SURF/forced_lc_by_id/{id}.dat', format='ascii')

    else:

        # Create lightcurve table with data from all files of the current ID
        for f in glob.glob(f'C:/Users/thuwa/Coding/SURF/forced_lc/{id}*.dat'):

            filt = f.split('_')[-1].split('.')[0]       # ZTF band
            field = f.split('_')[-2]                    # field

            # Create the table if first iteration
            if lc is None:      
                lc = Table(names=cols.split(','), dtype=dtypes.split(',')) 

            # Add rows to the table from the dat files. Discard rows with NaN
            try:
                tab = Table.read(f, format='ascii')
                for mjd, mag, magerr in tab[['col1','col2','col3']]:
                    if np.sum(np.isnan([mag, magerr]))==0:
                        lc.add_row([mjd, mag, magerr, filt, field])
            except Exception as e:
                continue #print(e)s
        
        # Save the processed light curve data
        if lc!=None:
            ascii.write(lc, f'C:/Users/thuwa/Coding/SURF/forced_lc_by_id/{id}.dat', overwrite=True)
    
    if lc!=None:

        if len(lc)>0:

            timeseries=dict()
            data=dict()
            dataerr=dict()

            for f in np.unique(lc['filt']):
                mjd=np.array([])
                flux=np.array([])
                fluxerr=np.array([])

                for field in np.unique(lc['field']):
                    
                    mask = (lc['filt'] == f)*(lc['field']==field)
                    cur_mjd = lc['jd'][mask] - 2400000.5
                    cur_flux = 3631e6*10**(-0.4*lc['mag'][mask])
                    cur_fluxerr = cur_flux*0.4*np.log(10)*lc['magerr'][mask]
                   
                    zp = np.nansum((cur_flux/cur_fluxerr**2)[cur_mjd<58500])/np.nansum((1/cur_fluxerr**2)[cur_mjd<58500]) 
                    if not np.isfinite(zp):
                        zp = 0
                    
                    mjd=np.append(mjd,cur_mjd)
                    flux=np.append(flux, cur_flux-zp)
                    fluxerr=np.append(fluxerr,cur_fluxerr)
                
                timeseries[f]=mjd
                data[f]=flux
                dataerr[f]=fluxerr
                

            LC=lightcurve(timeseries, data , dataerr, id)
            LC.regress()
            # #LC.plot()
            # #LC.findflare()
            LC.plot(show=True)
                