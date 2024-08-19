from lightcurveprocessor import LightCurve
import copy
from constants import *

# class Simulation(LightCurve):

#     def __init__()
def gaussian_rise(simulated_data: dict, timeseries: dict, t_start: dict, t_peak: float, peak_flux_ref: float, T0: float, sigma_rise: float):

    for filter in simulated_data.keys():
        for (i, t) in enumerate(timeseries[filter]):
            if (t-t_start[filter])<=t_peak and (t-t_start[filter])>=0:
                simulated_data[filter][i]+=peak_flux_ref*(B(filter,T0)/B(ref,T0))*np.exp(-(t-(t_peak+t_start[filter]))**2/(2*sigma_rise**2))

    return simulated_data

def power_law_rise(simulated_data: dict, timeseries: dict, t_start: dict, t_peak: float, peak_flux_ref: float, T0: float, t_fl: float, n: float):

    for filter in simulated_data.keys():
        for (i, t) in enumerate(timeseries[filter]):
            if (t-t_start[filter])<=t_peak and (t-t_start[filter])>=t_fl:
                simulated_data[filter][i]+=peak_flux_ref*(B(filter,T0)/B(ref,T0))*((t-(t_fl+t_start[filter]))/(t_peak-(t_fl+t_start[filter])))**n

    return simulated_data

def exponential_decline(simulated_data: dict, timeseries: dict, t_start: dict,t_peak: float, peak_flux_ref: float, T0: float, t_decay: float, t_plateau : float = np.inf):

    for filter in simulated_data.keys():
        for (i, t) in enumerate(timeseries[filter]):
            if (t-t_start[filter])>=t_peak and (t-t_start[filter])<=t_plateau:
                simulated_data[filter][i]+=peak_flux_ref*(B(filter,T0)/B(ref,T0))*np.exp(-(t-(t_peak+t_start[filter]))/t_decay)

    return simulated_data

def power_law_decline():
    pass


def model(data: dict , timeseries: dict, rise: str, decline :str, t_peak: float, peak_flux_ref: float, T0: float, **kwargs):

    simulated_data = copy.deepcopy(data)

    if rise=='r1':

        t_start=dict()
        for filter in simulated_data.keys():
            t_start[filter]=timeseries[filter][0]

        simulated_data = gaussian_rise(simulated_data, timeseries, t_start, t_peak, peak_flux_ref, T0, 
                                       kwargs['sigma_rise'])
    elif rise=='r2':

        t_start=dict()
        for filter in simulated_data.keys():
            t_start[filter]=timeseries[filter][0]

        simulated_data = power_law_rise(simulated_data, timeseries, t_start, t_peak, peak_flux_ref, T0,
                                        kwargs['t_fl'], kwargs['n'])
    else:
        print("Invalid rise model.")
        return

    if decline=='d1':
        
        simulated_data = exponential_decline(simulated_data, timeseries, t_start, t_peak, peak_flux_ref, T0,
                                             kwargs['t_decay'])
        
    elif decline=='d2':
         
        power_law_decline()

    elif decline=='d3':
        
        simulated_data = exponential_decline(simulated_data, timeseries, t_start, t_peak, peak_flux_ref, T0,
                                             kwargs['t_decay'], kwargs['t_plateau'])
        
    elif decline=='d4':
        pass
    elif decline=='d5':
        
        simulated_data = exponential_decline(simulated_data, timeseries, t_start, t_peak, peak_flux_ref, T0,
                                             kwargs['t_decay'])
        simulated_data = gaussian_rise(simulated_data, timeseries, 
                                       kwargs['t_start_secondary'], kwargs['t_peak_secondary'], kwargs['peak_flux_ref_secondary'], kwargs['T0_secondary'], kwargs['sigma_rise_secondary'])
        simulated_data = exponential_decline(simulated_data, timeseries, 
                                             kwargs['t_start_secondary'], kwargs['t_peak_secondary'], kwargs['peak_flux_ref_secondary'], kwargs['T0_secondary'], kwargs['t_decay_secondary'])

    elif decline=='d6':
        pass
    else:
        print("Invalid decline model.")
        return 
