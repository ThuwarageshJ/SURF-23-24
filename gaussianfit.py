import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class lightcurve:

    def __init__(self, timeseries, data, field, dataerr, name):
        self.timeseries=timeseries
        self.data=data
        self.dataerr=dataerr
        #self.type=type
        self.time_prediction=np.array([])
        self.mean_prediction=np.array([])
        self.std_prediction=np.array([])
        self.field=field
        self.flare=[]
        self.name=name
    
    def plot(self,mask= False, type='flux'):
        # zp = np.nansum((self.data/self.dataerr**2)[timeseries<58500])/np.nansum((1/fluxerr**2)[mjd<58500]) 
        # if not np.isfinite(zp):
        #     zp = 0
        plt.clf()
        plt.errorbar(self.timeseries, self.data, self.dataerr, fmt='o',  c=dict(zg="royalblue", zr="crimson")[self.field], label=self.field)
        plt.plot(self.time_prediction, self.mean_prediction, label="Mean prediction ")
        if len(self.mean_prediction)!=0:
            plt.fill_between(
                self.time_prediction.ravel(),
                self.mean_prediction - 1.96 * self.std_prediction,
                self.mean_prediction + 1.96 * self.std_prediction,
                alpha=0.5,
                label=r"95% confidence interval",
            )
        for i in range(len(self.flare)):
            print(self.flare[i])
            plt.axvspan(self.flare[i][0], self.flare[i][1], alpha=0.5, color='r')
        plt.legend()
        plt.xlabel('MJD')
        plt.ylabel('Flux [uJy]')
        plt.savefig(f'C:/Users/thuwa/Coding/SURF/samples/{self.name}.png')
        #plt.show()
        plt.close()

    def regress(self):
        if len(self.data)==0:
            return
        kernel = 1 * Matern(nu=1.5, length_scale=1, length_scale_bounds=(1e-9, 1e9))
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=np.square(self.dataerr) )
        gaussian_process.fit(self.timeseries.reshape(-1,1), self.data)
        self.time_prediction=np.linspace(self.timeseries[0], self.timeseries[-1], int((self.timeseries[-1]-self.timeseries[0])/2)).reshape(-1,1)
        self.mean_prediction, self.std_prediction = gaussian_process.predict(self.time_prediction, return_std=True)
    
    def calcraw(self, flux_diff):
        raw=0
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
        if len(self.mean_prediction)==0:
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
        for (i,t) in enumerate(self.time_prediction.ravel()):
            if i==0: continue
            dt=self.time_prediction.ravel()[i]-self.time_prediction.ravel()[i-1]
            raw=self.calcraw(self.mean_prediction[i]-self.mean_prediction[i-1])
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
            self.flare=flare
        
        plt.plot(times, raws, label='raw')
        plt.plot(times, levels, label='level')
        plt.legend()
        #plt.show()
        plt.close()
