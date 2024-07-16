import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import TheilSenRegressor

class lightcurve:

    def __init__(self, timeseries: dict, data: dict , dataerr: dict, id: str)-> None:

        self.alpha = .3
        self.T=20

        self.timeseries=timeseries
        self.data=data
        self.dataerr=dataerr

        self.filters=self.data.keys()

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
            self.thiel_sen_parameters[filter]=None
            self.gp_parameters[filter]=None

        self.regress()
        #self.find_thiel_sen()

        self.id =id
        # self.regressed=False

    def plot(self, save_loc: str = None, regress_before_plotting : bool = True, show: bool =False, save: bool = False)-> None:
        
        if self.null: 
            return
        
        #plt.clf()
        fig = plt.figure(figsize=(10/1.6*3,10/1.6))
        ax = fig.subplots()

        for filter in self.filters:

            ax.errorbar(self.timeseries[filter], self.data[filter], self.dataerr[filter], fmt='o',  c=dict(zg="royalblue", zr="crimson")[filter], label=filter)
            
            if len(self.mean_prediction[filter])!=0:
                ax.plot(self.time_prediction[filter], self.mean_prediction[filter], label="Mean prediction "+filter)
                ax.fill_between(
                    self.time_prediction[filter],
                    self.mean_prediction[filter] - 1.96 * self.std_prediction[filter],
                    self.mean_prediction[filter] + 1.96 * self.std_prediction[filter],
                    alpha=0.5,
                    label=r"95% confidence interval "+filter,
                )
            
            #ax.plot(self.time_prediction[filter], self.thiel_sen_prediction[filter], label="Thiel Sen Line "+filter)
            
            for i in range(len(self.flares_t[filter])):
                plt.axvspan(self.flares_t[filter][i][0],self.flares_t[filter][i][1], alpha=0.35, color=dict(zg='b', zr='r')[filter])

        plt.legend()
        plt.grid()
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
            kernel = 1 * Matern(nu=1.5, length_scale=1, length_scale_bounds=(1e-9, 1e9))
            gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=np.square(self.dataerr[filter]) )
            gaussian_process.fit(self.timeseries[filter].reshape(-1,1), self.data[filter])

            self.gp_parameters[filter]=gaussian_process.kernel_

            self.mean_prediction[filter], self.std_prediction[filter] = gaussian_process.predict(self.time_prediction[filter].reshape(-1,1), return_std=True)

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

    def find_flare(self, user: bool = False, reset_params: bool = True):

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
                raw=self.calcraw(self.mean_prediction[filter][i]-self.mean_prediction[filter][i-1], self.std_prediction[filter][i])
                level=level+dt*(raw-level)/self.T

                if level>0.8 and not flare_began:
                    flare_began=True
                    flare_temp=(t, i)
                elif level<-0.8 and flare_began:
                    flare_began=False
                    timing=(flare_temp[0], t)
                    loc=(flare_temp[1], i)
                    flare_t.append(timing)
                    flare_loc.append(loc)
            
            self.flares_t[filter]=flare_t
            self.flares_loc[filter]=flare_loc

        self.find_peak_times()
        
        self.plot(show=True)

    def find_peak_times(self):

        if self.null:
            return

        for filter in self.filters:

            half_to_peak=[]
            peak_to_half=[]

            for loc in self.flares_loc[filter]:

                peak_idx = np.argmax(self.mean_prediction[filter][loc[0]:loc[1]+1])+loc[0]

                peak_flux = self.mean_prediction[filter][peak_idx]
                peak_t = self.time_prediction[filter][peak_idx]

                for i in range(peak_idx, -1, -1):
                    if self.mean_prediction[filter][i]<peak_flux/2:
                        half_peak_t = (self.time_prediction[filter][i]+self.time_prediction[filter][i+1])/2
                        half_to_peak.append(peak_t - half_peak_t)
                        break
                
                for i in range(peak_idx, len(self.time_prediction[filter])):
                    if self.mean_prediction[filter][i]<peak_flux/2:
                        half_peak_t = (self.time_prediction[filter][i]+self.time_prediction[filter][i-1])/2
                        peak_to_half.append(half_peak_t - peak_t)
                        break
                        
            self.half_to_peak[filter]=half_to_peak
            self.peak_to_half[filter]=peak_to_half
        





            

            
        
