import os, glob
import numpy as np

cur_folder_path=os.path.dirname(__file__)   # current folder path
print(cur_folder_path)

"""
    Folder paths and program mode variables
"""
cores = 4                               # no. of cores for multi processing
batch_size = 100                        # batch size for multi processing
use_multiprocessing = 0             # whether to use multi processing

lc_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\forced_lc'               # folder for light curve files
fig_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\samples'                # folder to save plots
pickle_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\pickles'             # folder to save processed light curves as pickle files     
temp_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\forced_lc_by_id'       # ignore
simulations_path = os.path.join(cur_folder_path, 'simulations') 
simulations_raw_fig_path = os.path.join(simulations_path, 'raw') 
simulations_fig_path =  os.path.join(simulations_path, 'fig')  
simulations_pickle_path =  os.path.join(simulations_path, 'pickles') 

show= False                             # show plots in a window after processing 
save= True                              # save plots after processing
save_pickle = False                     # save pickles after processing

adjust_parameters=False                 # if True, will be prompted to a command line UI to play around T and alpha values for the detector
reset_params = True                     # if adjusting T and alpha, reset them to their default values after adjusting
plot_std= False                         # plot 95% confidence interval from GP regression
print_flare_parameters = False          # print flare physical parameters after processing light curve
temp_store=False                        # ignore

"""
    Default parameters and datatypes for processing the light curves
"""
alpha = .2                              # a measure of lower bound for slope of the flare. Decrease to detect slowly rising flares
T=9                                   # a measure of no. of days of continuous flux increase to be detected as a flare. Decreast to detect short flares.
post_peak_g_r_days=5                    # no. of time data points after peak to calculate for g-r color
prediction_interval = 2                 # time interval for the timeseries on to which GP fit is done
N_low=20                                # no. of lowest data points used for zero point calculation. Ignore for now
mjd_zp = 58500                          # mjd used for zero point calculation for each field

"""
    Simulation parameters
"""
peak_flux_ref_range =[0, 151]
T0_range = [1e3, 1e7+1]
time_range = [2, 366]
n_sims_per_sample = 5                   # no. of simulations to be created from a flat light curve file
T0_avg = 21000
sigma_rise_avg = 20

"""
    Global constants: No need to change
"""
h=6.626e-34

c=3e8
k=1.38e-23
frequencies={
    'zg':6.3e14,
    'zr':4.3e14
}
ref='zg'

def B(filt,T):
    v=frequencies[filt]
    pow=h*v/(k*T)
    return 2*h*v**3/(c**2*(np.exp(pow)-1))

pickle_paths=[os.path.join(pickle_path, 'positives'), os.path.join(pickle_path, 'negatives')]
fig_paths=[os.path.join(fig_path, 'positives'), os.path.join(fig_path, 'negatives')]
sim_pickle_paths=[os.path.join(simulations_pickle_path, 'positives'), os.path.join(simulations_pickle_path, 'negatives')]
sim_fig_paths=[os.path.join(simulations_fig_path, 'positives'), os.path.join(simulations_fig_path, 'negatives')]
sim_raw_fig_paths=[os.path.join(simulations_raw_fig_path, 'positives'), os.path.join(simulations_raw_fig_path, 'negatives')]

mjd_adjustment = 2400000.5

# Columns and corresponding datatypes for lightcurves.
cols = 'jd,mag,magerr,filt,field,flux,fluxerr,adjflux'
dtypes = 'int,float,float,str,int,float,float,float'

cols1 = 'jd,filt,fluxerr,adjflux'
dtypes1 = 'int,str,float,float'

# Run to test folder path validity
if __name__=="__main__":

    if not glob.glob(f'{lc_path}\*.dat'):
        print("Provided path for lightcurves doesn't have any data files")
    elif not os.path.exists(lc_path):
        print("Provided path for lightcurves doesn't exist")
    else:
        print("Light curve path OK")

    if not os.path.exists(fig_path):
        print("Provided path for saving plots doesn't exist")
    else:
        print("Plots saving path OK")

    if not os.path.exists(pickle_path):
        print("Provided path for saving pickles doesn't exist")
    else:
        print("Pickles saving path OK")

    if temp_store:
        if not os.path.exists(temp_path):
            print("Provided path for saving temporary files doesn't exist")
        else:
            print("Temp files saving path OK")
