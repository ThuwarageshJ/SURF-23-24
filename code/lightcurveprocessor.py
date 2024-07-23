import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import TheilSenRegressor
import george
from george.kernels import Matern32Kernel
import scipy.optimize as op
import pickle
import os

class LightCurve:

    def __init__(self, timeseries: dict, data: dict , dataerr: dict, id: str)-> None:

        self.alpha = .23
        self.T=9

        self.timeseries=timeseries
        self.data=data
        self.dataerr=dataerr

        self.filters=list(self.data.keys())

        self.N=[]

        self.null = True

        for filter in self.filters:
            self.N.append(len(self.timeseries[filter]))
            if len(self.timeseries[filter]!=0): 
                self.null = False

        self.time_prediction=dict()

        self.mean_prediction=dict()
        self.std_prediction=dict()
        self.gp_parameters=dict()

        self.thiel_sen_prediction=dict()
        self.thiel_sen_parameters=dict()

        self.flares_t=dict()
        self.flares_loc=dict()

        self.peaks=dict()
        self.peaks_loc=dict()
        self.peaks_err=dict()

        self.g_r_color_at_peak=dict()
        self.g_r_color_at_peak_err=dict()

        self.g_r_color_change_rates=dict()
        self.g_r_color_change_rates_err=dict()

        self.half_to_peak=dict()
        self.peak_to_half=dict()

        self.t_start=np.inf
        self.t_end=-np.inf

        if not self.null:
            for filter in self.filters:
                if len(self.timeseries[filter]!=0):
                    self.t_start=min(self.t_start, self.timeseries[filter][0])
                    self.t_end=max(self.t_end, self.timeseries[filter][-1])
        else:
            print('Empty dataset!')

        for filter in self.filters:
            
            self.time_prediction[filter]=np.linspace(self.t_start, self.t_end, int((self.t_end-self.t_start)/2)) if len(self.timeseries[filter])!=0 else np.array([])
            self.mean_prediction[filter]=np.array([])
            self.std_prediction[filter]=np.array([])
            self.thiel_sen_prediction[filter]=np.array([])
            self.flares_t[filter]=np.array([])
            self.flares_loc[filter]=np.array([])
            self.half_to_peak[filter]=np.array([])
            self.peak_to_half[filter]=np.array([])
            self.peaks[filter]=np.array([])
            self.peaks_loc[filter]=np.array([])
            self.peaks_err[filter]=np.array([])
            self.g_r_color_at_peak[filter]=np.array([])
            self.g_r_color_change_rates[filter]=np.array([])
            self.g_r_color_at_peak_err[filter]=np.array([])
            self.g_r_color_change_rates_err[filter]=np.array([])
            self.thiel_sen_parameters[filter]=None
            self.gp_parameters[filter]=None

        self.regress()
        #self.find_thiel_sen()

        self.id =id

    def plot(self, save_loc: str = None, plot_thiel_sen: bool =False, plot_std: bool = False, plot_data: bool = True, show: bool =False, save: bool = False)-> None:
        
        if self.null: 
            return
        
        #plt.clf()
        fig = plt.figure(figsize=(10/1.6*3,10/1.6))
        ax = fig.subplots()

        for filter in self.filters:

            if plot_data:
                ax.errorbar(self.timeseries[filter], self.data[filter], self.dataerr[filter], fmt='o',  c=dict(zg="royalblue", zr="crimson")[filter], label=filter)
            
            if len(self.mean_prediction[filter])!=0:
                ax.plot(self.time_prediction[filter], self.mean_prediction[filter], label="Mean prediction "+filter)

                if plot_std:
                    ax.fill_between(
                        self.time_prediction[filter],
                        self.mean_prediction[filter] - 1.96 * self.std_prediction[filter],
                        self.mean_prediction[filter] + 1.96 * self.std_prediction[filter],
                        alpha=0.5,
                        label=r"95% confidence interval "+filter,
                )
            
            if len(self.thiel_sen_prediction[filter])!=0 and plot_thiel_sen:
                ax.plot(self.time_prediction[filter], self.thiel_sen_prediction[filter], label="Thiel Sen Line "+filter)
            
            for i in range(len(self.flares_t[filter])):
                plt.axvspan(self.flares_t[filter][i][0],self.flares_t[filter][i][1], alpha=0.35, color=dict(zg='b', zr='r')[filter])

        plt.legend()
        #plt.grid()
        plt.xlabel('MJD')
        plt.ylabel('Flux [uJy]')
        _=plt.title(self.id+' T='+str(self.T)+' alpha='+str(self.alpha))

        if save and save_loc is not None:
            plt.savefig(f'{save_loc}/{self.id}.png')

        if show:
            plt.show()

        plt.close()

    def regress(self):

        if self.null:
            return

        for filter in self.filters:
            if len(self.data[filter])==0:
                continue

        #     kernel = 1 * Matern(nu=1.5, length_scale=1, length_scale_bounds=(1e-9, 1e9))
        #     gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=np.square(self.dataerr[filter]) )
        #     gaussian_process.fit(self.timeseries[filter].reshape(-1,1), self.data[filter])

        #     self.gp_parameters[filter]=gaussian_process.kernel_

        #     self.mean_prediction[filter], self.std_prediction[filter] = gaussian_process.predict(self.time_prediction[filter].reshape(-1,1), return_std=True)

            kernel = 10*Matern32Kernel(1e5)

            gp = george.GP(kernel)
            #gp.compute(self.timeseries[filter], self.dataerr[filter])

           # self.gp_parameters[filter]=gaussian_process.kernel_
           # Define the objective function (negative log-likelihood in this case).
            def nll(p):
                # Update the kernel parameters and compute the likelihood.
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(self.data[filter], quiet=True)
                return -ll if np.isfinite(ll) else 1e25

            # And the gradient of the objective function.
            def grad_nll(p):
                # Update the kernel parameters and compute the likelihood.
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(self.data[filter], quiet=True)
            
            gp.compute(self.timeseries[filter], self.dataerr[filter])

            # Print the initial ln-likelihood.
            #print(gp.lnlikelihood(self.data[filter]))

            # Run the optimization routine.
            p0 = gp.get_parameter_vector()
            results = op.minimize(nll, p0, jac=grad_nll)

            # Update the kernel and print the final log-likelihood.
            gp.set_parameter_vector(results.x)
            #print(gp.lnlikelihood(self.data[filter]))

            mu, cov= gp.predict(self.data[filter],self.time_prediction[filter])
            self.mean_prediction[filter]=mu
            self.std_prediction[filter] =np.sqrt(np.diag(cov))

    def find_thiel_sen(self):

        if self.null:
            return

        for filter in self.filters:
            if len(self.data[filter])==0:
                continue
            thiel_sen=TheilSenRegressor(random_state=42)   # recheck random state
            thiel_sen.fit(self.timeseries[filter].reshape(-1,1), self.data[filter])
            self.thiel_sen_prediction[filter]=thiel_sen.predict(self.time_prediction[filter].reshape(-1,1))

            self.thiel_sen_parameters[filter] = [thiel_sen.coef_, thiel_sen.intercept_]
            
    def calcraw(self, flux_diff: float, err: float)-> float:

        raw=0
        flux_diff=flux_diff/err

        if flux_diff>self.alpha:
            raw=1
        elif flux_diff>self.alpha/3:
            raw=0.5
        elif flux_diff>self.alpha/6:
            raw=0.3
        elif flux_diff<-self.alpha:
            raw=-1
        elif flux_diff<-self.alpha/3:
            raw=-0.5
        elif flux_diff<-self.alpha/6:
            raw=-.3
        
        return raw

    def find_flare(self, user: bool = False, show_plot: bool = True, reset_params: bool = True):

        if self.null:
            return
        
        if user:

            original_T = self.T
            original_alpha = self.alpha

            while True:

                params = input('Input T and alpha. x for return')

                if params=='x':
                    break
                else:
                    params= tuple(float(i) for i in params.split())

                (self.T, self.alpha) = params[0], params[1]

                self.exponential_filter()

                self.plot(show=show_plot)

            if reset_params:
                self.T = original_T
                self.alpha = original_alpha
            
        else:

            self.exponential_filter()

    def exponential_filter(self):

        if self.null:
            return
        
        for filter in self.filters:

            if len(self.mean_prediction[filter])==0:
                continue
            
            flare_began=False
            flare_t=[]
            flare_loc=[]
            level=0

            for (i,t) in enumerate(self.time_prediction[filter]):

                if i==0: continue

                dt=self.time_prediction[filter][i]-self.time_prediction[filter][i-1]
                raw=self.calcraw(self.mean_prediction[filter][i]-self.mean_prediction[filter][i-1], np.sqrt(self.std_prediction[filter][i]**2+self.std_prediction[filter][i-1]**2))
                level=level+dt*(raw-level)/self.T

                if level>0.8 and not flare_began:
                    flare_began=True
                    flare_temp=(t, i)
                elif (level<-0.8 or i==len(self.time_prediction[filter])-1) and flare_began:    # add increasing 
                    flare_began=False
                    timing=(flare_temp[0], t)
                    loc=(flare_temp[1], i)
                    flare_t.append(timing)
                    flare_loc.append(loc)
            
            self.flares_t[filter]=flare_t
            self.flares_loc[filter]=flare_loc

        self.find_flare_parameters()

    def find_flare_parameters(self):

        if self.null:
            return

        for filter in self.filters:

            half_to_peak=[]
            peak_to_half=[]
            peaks=[]
            peaks_loc=[]
            peaks_err=[]
            g_r_color=[]
            g_r_color_err=[]
            g_r_color_changes=[]
            g_r_color_changes_err=[]

            for loc in self.flares_loc[filter]:

                peak_idx = np.argmax(self.mean_prediction[filter][loc[0]:loc[1]+1])+loc[0]

                peak_flux=(self.mean_prediction['zg'][peak_idx], self.mean_prediction['zr'][peak_idx])
                peak_flux_err=(self.std_prediction['zg'][peak_idx], self.std_prediction['zr'][peak_idx])
                g_r=-2.5*np.log10(peak_flux[0]/peak_flux[1])
                g_r_err=2.5*np.sqrt((peak_flux_err[0]/peak_flux[0])**2+(peak_flux_err[1]/peak_flux[1])**2)/np.log(10)

                g_r_changes=[]
                g_r_changes_err=[]

                for i in range(peak_idx+1, min(peak_idx+6, len(self.time_prediction[filter]))):
                    val=(-2.5*(np.log10(self.mean_prediction['zg'][i]*self.mean_prediction['zr'][i-1]/(self.mean_prediction['zr'][i]*self.mean_prediction['zg'][i-1]))))/(self.time_prediction[filter][i]-self.time_prediction[filter][i-1])
                    frac=self.mean_prediction['zg'][i]*self.mean_prediction['zr'][i-1]/(self.mean_prediction['zr'][i]*self.mean_prediction['zg'][i-1])
                    frac_err=0
                    for j in range(i-1,i+1):
                        frac_err+=self.std_prediction['zg'][j]**2/self.mean_prediction['zg'][j]**2+self.std_prediction['zr'][j]**2/self.mean_prediction['zr'][j]**2
                    g_r_changes.append(val)
                    g_r_changes_err.append(abs(val)*np.sqrt((frac_err/(np.log(10)*np.log10(frac)))**2))

                peaks_loc.append(peak_idx)
                g_r_color_changes.append(tuple(g_r_changes))
                g_r_color_changes_err.append(tuple(g_r_changes_err))
                g_r_color.append(g_r)
                g_r_color_err.append(g_r_err)
                peaks.append(peak_flux)
                peaks_err.append(peak_flux_err)

                peak_t = self.time_prediction[filter][peak_idx]

                j=0 if filter=='zg' else 1

                for i in range(peak_idx-1, -1, -1):
                    if self.mean_prediction[filter][i]<peak_flux[j]/2:
                        half_peak_t = (self.time_prediction[filter][i]+self.time_prediction[filter][i+1])/2
                        half_to_peak.append(peak_t - half_peak_t)
                        break
                    if i==0:
                        half_to_peak.append(None)
                
                for i in range(peak_idx, len(self.time_prediction[filter])):
                    if self.mean_prediction[filter][i]<peak_flux[j]/2:
                        half_peak_t = (self.time_prediction[filter][i]+self.time_prediction[filter][i-1])/2
                        peak_to_half.append(half_peak_t - peak_t)
                        break
                    if i==len(self.time_prediction[filter]-1):
                        peak_to_half.append(None)
                        
            self.half_to_peak[filter]=half_to_peak
            self.peak_to_half[filter]=peak_to_half
            self.peaks[filter]=peaks
            self.peaks_loc[filter]=peaks_loc
            self.peaks_err[filter]=peaks_err
            self.g_r_color_at_peak[filter]=g_r_color
            self.g_r_color_at_peak_err[filter]=g_r_color_err
            self.g_r_color_change_rates[filter]=g_r_color_changes
            self.g_r_color_change_rates_err[filter]=g_r_color_changes_err


            if self.flares_loc[filter]:
                print(filter)
                print(' Half peak to peak times: ', self.half_to_peak[filter], ' days')
                print(' Peak to half peak times: ', self.peak_to_half[filter], 'days')
                print(' Peak fluxes: (ZG, ZR): ', self.peaks[filter], ' uJy')
                print(' g-r colors at peak: ', self.g_r_color_at_peak[filter], ' mag')
                print(' g-r color change rates for 5 days after peak: ', self.g_r_color_change_rates[filter], ' mag day-1')

    
