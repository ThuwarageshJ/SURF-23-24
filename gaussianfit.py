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
        self.flares=dict()

        t_start=np.inf
        t_end=-np.inf

        if not self.null:
            for filter in self.filters:
                if len(self.timeseries[filter]!=0):
                    t_start=min(t_start, self.timeseries[filter][0])
                    t_end=max(t_end, self.timeseries[filter][-1])
        else:
            print('Empty dataset!')

        for filter in self.filters:
            
            self.time_prediction[filter]=np.linspace(t_start, t_end, int((t_end-t_start)/2)).reshape(-1,1) if len(self.timeseries[filter])!=0 else np.array([])
            self.mean_prediction[filter]=np.array([])
            self.std_prediction[filter]=np.array([])
            self.thiel_sen_prediction[filter]=np.array([])
            self.flares[filter]=np.array([])
            self.thiel_sen_parameters[filter]=None
            self.gp_parameters[filter]=None

        self.regress()
        self.find_thiel_sen()

        # # #self.flare=[]

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
                    self.time_prediction[filter].ravel(),
                    self.mean_prediction[filter] - 1.96 * self.std_prediction[filter],
                    self.mean_prediction[filter] + 1.96 * self.std_prediction[filter],
                    alpha=0.5,
                    label=r"95% confidence interval "+filter,
                )
            
            ax.plot(self.time_prediction[filter], self.thiel_sen_prediction[filter], label="Thiel Sen Line "+filter)
            
            for i in range(len(self.flares[filter])):
                plt.axvspan(self.flares[filter][i][0],self.flares[filter][i][1], alpha=0.35, color=dict(zg='b', zr='r')[filter])

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

        # self.regressed=True
        if self.null:
            return

        for filter in self.filters:
            if len(self.data[filter])==0:
                continue
            kernel = 1 * Matern(nu=1.5, length_scale=1, length_scale_bounds=(1e-9, 1e9))
            gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=np.square(self.dataerr[filter]) )
            gaussian_process.fit(self.timeseries[filter].reshape(-1,1), self.data[filter])

            self.gp_parameters[filter]=gaussian_process.kernel_

            self.mean_prediction[filter], self.std_prediction[filter] = gaussian_process.predict(self.time_prediction[filter], return_std=True)

    def find_thiel_sen(self):

        if self.null:
            return

        for filter in self.filters:
            if len(self.data[filter])==0:
                continue
            thiel_sen=TheilSenRegressor(random_state=42)   # recheck random state
            thiel_sen.fit(self.timeseries[filter].reshape(-1,1), self.data[filter])
            self.thiel_sen_prediction[filter]=thiel_sen.predict(self.time_prediction[filter])

            self.thiel_sen_parameters[filter] = [thiel_sen.coef_, thiel_sen.intercept_]
            
    def calcraw(self, flux_diff, err):

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

    def findflare(self):

        if self.null:
            return
        
        while True:

            params = input('Input T and alpha. x for return')

            if params=='x':
                break
            else:
                params= tuple(float(i) for i in params.split())

            (self.T, self.alpha) = params[0], params[1]

            for filter in self.filters:

                if len(self.mean_prediction[filter])==0:
                    return
                
                flare_began=False
                flare=[]
                level=0

                for (i,t) in enumerate(self.time_prediction[filter].ravel()):

                    if i==0: continue

                    dt=self.time_prediction[filter].ravel()[i]-self.time_prediction[filter].ravel()[i-1]
                    raw=self.calcraw(self.mean_prediction[filter][i]-self.mean_prediction[filter][i-1], self.std_prediction[filter][i])
                    level=level+dt*(raw-level)/self.T

                    if level>0.8 and not flare_began:
                        flare_began=True
                        flare_temp=t
                    elif level<-0.8 and flare_began:
                        flare_began=False
                        timing=(flare_temp, t)
                        flare.append(timing)
                
                if flare:
                    self.flares[filter]=flare
        
            self.plot(show=True)

            

            
        
