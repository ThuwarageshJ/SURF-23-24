import os, glob

"""
    Global constants: No need to change
"""
cur_folder_path=os.path.dirname(__file__)
mjd_adjustment = 2400000.5
mjd_zp = 58500

"""
    Folder paths and program mode variables
"""

batch_size = 100
lc_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\forced_lc'
fig_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\samples_1'
pickle_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\pickles_1'
temp_path = r'C:\Users\thuwa\Coding\SURF\SURF-23-24\code\forced_lc_by_id'
temp_store=False

show= False
save= True
save_pickle = True
adjust_parameters=False
reset_params = True
plot_std=False
print_flare_parameters = False

"""
    Default parameters and datatypes for processing the light curves
"""
alpha = .2
T=9
post_peak_g_r_days=5
prediction_interval = 2
N_low=20

# Columns and corresponding datatypes for lightcurves.
cols = 'jd,mag,magerr,filt,field,flux,fluxerr,adjflux'
dtypes = 'int,float,float,str,int,float,float,float'


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
