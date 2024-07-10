# %reload_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sys, os, glob
from astropy.table import Table
from tqdm import *
from gaussianfit import lightcurve

cols = 'jd,mag,magerr,filt,field'
dtypes = 'int,float,float,str,int'
ids = np.unique([f.split('\\')[-1].split('_')[0] for f in glob.glob('C:/Users/thuwa/Coding/SURF/SURF-23-24/forced_lc/*.dat')])
i=0
#ids=['103661510464503659']
for id in tqdm(ids):
    #print(id)
    lc = None
    for f in glob.glob(f'C:/Users/thuwa/Coding/SURF/SURF-23-24/forced_lc/{id}*.dat'):
        filt = f.split('_')[-1].split('.')[0]
        field = f.split('_')[-2]
        if lc is None:
            lc = Table(names=cols.split(','), dtype=dtypes.split(','))
            
        try:
            tab = Table.read(f, format='ascii')
            for mjd, mag, magerr in tab[['col1','col2','col3']]:
                lc.add_row([mjd, mag, magerr, filt, field])
        except Exception as e:
            continue #print(e)

    if lc!=None:
        if len(lc)>0:
            plt.figure(figsize=(10/1.6*3,8/1.6))
            for f in np.unique(lc['filt']):
                for field in np.unique(lc['field']):
                    #print(f, field)
                    mask = (lc['filt'] == f)*(lc['field']==field)
                    mjd = lc['jd'][mask] - 2400000.5
                    flux = 3631e6*10**(-0.4*lc['mag'][mask])
                    fluxerr = flux*0.4*np.log(10)*lc['magerr'][mask]
                    #print(len(flux))
                    ############################
                    zp = np.nansum((flux/fluxerr**2)[mjd<58500])/np.nansum((1/fluxerr**2)[mjd<58500]) 
                    if not np.isfinite(zp):
                        zp = 0
                    print(f, field)
                    print(flux, fluxerr)
                    #######################################
                    # LC=lightcurve(np.array(mjd), np.array(flux-zp), f, np.array(fluxerr))
                    # LC.regress()
                    # #LC.plot()
                    # LC.findflare()
                    # LC.plot()
                    #######################################
            #         plt.errorbar(mjd, flux-zp, yerr=fluxerr, fmt='o', c=dict(zg="royalblue", zr="crimson")[f])
            #         ################################
            #     plt.errorbar(np.nan, np.nan, yerr=0.1, fmt='o', c=dict(zg="royalblue", zr="crimson")[f], label=f)
                
            # plt.legend()
            # plt.xlabel('MJD')
            # plt.ylabel('Flux [uJy]')
            # plt.savefig(f'C:/Users/thuwa/Coding/SURF/raw.png')
            # #plt.show()
            # plt.close()