import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import TheilSenRegressor

class lightcurve:

    def __init__(self, timeseries: dict, data: dict , dataerr: dict, id: str)-> None:

        self.timeseries=timeseries
        self.data=data
        self.dataerr=dataerr

        self.filters=self.data.keys()

        self.time_prediction=dict()
        self.mean_prediction=dict()
        self.std_prediction=dict()
        self.thiel_sen_prediction=dict()
        self.flares=dict()

        for filter in self.filters:
            self.time_prediction[filter]=np.linspace( min(self.timeseries[filter]), max(self.timeseries[filter]), int((max(self.timeseries[filter])-min(self.timeseries[filter]))/2)).reshape(-1,1) if len(self.timeseries[filter])!=0 else np.array([])
            self.mean_prediction[filter]=np.array([])
            self.std_prediction[filter]=np.array([])
            self.thiel_sen_prediction[filter]=np.array([])
            self.flares[filter]=np.array([])

        self.regress()
        #self.find_thiel_sen()

        #self.flare=[]

        self.id =id
        self.regressed=False

    def plot(self, regress_before_plotting : bool = True, show: bool =False, save: bool = False)-> None:
    
        #plt.clf()
        plt.figure(figsize=(10/1.6*3,10/1.6))

        for filter in self.filters:
            plt.errorbar(self.timeseries[filter], self.data[filter], self.dataerr[filter], fmt='o',  c=dict(zg="royalblue", zr="crimson")[filter], label=filter)
            plt.plot(self.time_prediction[filter], self.mean_prediction[filter], label="Mean prediction "+filter)

            if len(self.mean_prediction[filter])!=0:
                plt.fill_between(
                    self.time_prediction[filter].ravel(),
                    self.mean_prediction[filter] - 1.96 * self.std_prediction[filter],
                    self.mean_prediction[filter] + 1.96 * self.std_prediction[filter],
                    alpha=0.5,
                    label=r"95% confidence interval",
                )
            
            #plt.plot(self.time_prediction[filter], self.thiel_sen_prediction[filter], label="Thiel Sen Line "+filter)
            


        for filter in self.filters:
            for i in range(len(self.flares[filter])):
                plt.axvspan(self.flares[filter][i][0],self.flares[filter][i][1], alpha=0.5, color='r')

        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Flux [uJy]')
        if save:
            plt.savefig(f'C:/Users/thuwa/Coding/SURF/samples/{self.id}.png')
        if show:
            plt.show()
        plt.close()

    def regress(self):

        self.regressed=True

        for filter in self.filters:
            if len(self.data[filter])==0:
                continue
            kernel = 1 * Matern(nu=1.5, length_scale=1, length_scale_bounds=(1e-9, 1e9))
            gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=np.square(self.dataerr[filter]) )
            gaussian_process.fit(self.timeseries[filter].reshape(-1,1), self.data[filter])

            
            self.mean_prediction[filter], self.std_prediction[filter] = gaussian_process.predict(self.time_prediction[filter], return_std=True)

    def find_thiel_sen(self):

        for filter in self.filters:
            if len(self.data[filter])==0:
                continue
            thiel_sen=TheilSenRegressor(random_state=42)
            thiel_sen.fit(self.timeseries[filter].reshape(-1,1), self.data[filter])
            self.thiel_sen_prediction[filter]=thiel_sen.predict(self.time_prediction[filter])

    def calcraw(self, flux_diff, err):
        raw=0
        flux_diff=flux_diff/err**2
        if flux_diff>.3:
            raw=1
        elif flux_diff>.1:
            raw=0.5
        elif flux_diff>0.05:
            raw=0.3
        elif flux_diff<-.3:
            raw=-1
        elif flux_diff<-.1:
            raw=-0.5
        elif flux_diff<0.05:
            raw=-.3
        
        return raw

    def findflare(self):

        for filter in self.filters:
            if len(self.mean_prediction[filter])==0:
                return
            flare_began=False
            flare_end=False
            flare=[]
            #dt=2
            T=20
            level=0
            raws=[]
            levels=[]
            times=[]
            for (i,t) in enumerate(self.time_prediction[filter].ravel()):
                if i==0: continue
                dt=self.time_prediction[filter].ravel()[i]-self.time_prediction[filter].ravel()[i-1]
                raw=self.calcraw(self.mean_prediction[filter][i]-self.mean_prediction[filter][i-1], self.std_prediction[filter][i])
                raws.append(raw)
                level=level+dt*(raw-level)/T
                levels.append(level)
                times.append(t)
                if level>0.8 and not flare_began:
                    flare_began=True
                    flare_temp=t
                elif level<-0.8 and flare_began:
                    flare_began=False
                    timing=(flare_temp, t)
                    flare.append(timing)
            
            if flare:
                self.flares[filter]=flare
        
        # plt.plot(times, raws, label='raw')
        # plt.plot(times, levels, label='level')
        # plt.legend()
        # #plt.show()
        # plt.close()
