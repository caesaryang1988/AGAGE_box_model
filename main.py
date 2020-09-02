#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:40:14 2020

@author: hy337
"""

# READ INPUT FILES


import numpy as np
import pandas as pd
import time
from datetime import datetime
from scipy.sparse import bsr_matrix,spdiags,eye
from scipy.linalg import lu_factor,lu_solve

def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

work_dir = '/Users/hy337/Study/Data_Science/Projects/AGAGE_box_model/'

sec2year = 1/365/24/3600
dt = 1/12
start_year = 1977
end_year = 2020
num_date = (end_year-start_year+1)*12
yymm = []
year_dec = []
for year in range(start_year,end_year+1):
    for month in range(1,13):
        yyyymm = str(year*100+month)
        yyyymm = datetime.strptime(yyyymm,"%Y%m")
        yymm.append(yyyymm)
        yyyymm = toYearFraction(yyyymm)
        year_dec.append(yyyymm)

oh = np.genfromtxt(work_dir+'agage_box_model_input_oh.csv',delimiter = ',',skip_header=1)
oh = oh[:,1:]
temp = np.genfromtxt(work_dir+'agage_box_model_input_temp.csv',delimiter = ',',skip_header=1)
temp = temp[:,1:]
dif = np.genfromtxt(work_dir+'agage_box_model_input_parameters_CFC_diffusion.csv',delimiter = ',',skip_header=1)
dif = dif[:,1:]
adv = np.genfromtxt(work_dir+'agage_box_model_input_parameters_CFC_advection.csv',delimiter = ',',skip_header=1)
adv = adv[:,1:]



Name = ['SF6','CFC11','CFC12','CFC113','HCFC22','HCFC141b','HCFC142b','HFC134a','HFC152a','CCl4']
Arr_A  = np.array([  1.0e-30, 1.0e-12,  1.0e-12,  1.0e-30,  1.05e-12, 1.25e-12,  1.3e-12, 1.05e-12,  8.7e-13,  1.0e-12 ])
Arr_ER = np.array([  1600   , 3700   ,  3600   ,  1600   ,  1600    , 1600    ,  1770   , 1630    ,  975    ,  2300    ])

cct_ltne = np.zeros((len(Name),num_date))
cct_ltne[:,:] = np.nan
for i in range(len(Name)):
    species = Name[i]
    filename = work_dir+'HATS_Data/'+species.lower()+'_197701-202012.csv'
    cct_tab = pd.read_csv(filename,index_col=0)
    cct_ltne = cct_tab.loc[:,['nwr','mhd']]
    cct_ltne = np.nanmean(cct_ltne.to_numpy(),1)
    run_time = cct_tab.index[np.argwhere(~np.isnan(cct_ltne))].tolist()
    
    XI0 = np.zeros((11,1))
    X1 = np.ones((1,len(run_time)))
    X1 = cct_ltne[~np.isnan(cct_ltne)]
    XIvec = np.empty((0,11))
    
    for t_idx in range(len(run_time)):
        t = str(run_time[t_idx])
        mo = int(t.split(sep='-')[1]) - 1
        
        k01 = 1/ dif[mo,0];  k10 = k01;
        k12 = 1/ dif[mo,1];  k21 = k12;
        k23 = 1/ dif[mo,2];  k32 = k23;
        k45 = 1/ dif[mo,3];  k54 = k45;
        k56 = 1/ dif[mo,4];  k65 = k56;
        k67 = 1/ dif[mo,5];  k76 = k67;
        k40 = 1/ dif[mo,6];  k04 = k40;
        k51 = 1/ dif[mo,7];  k15 = k51;
        k62 = 1/ dif[mo,8];  k26 = k62;
        k73 = 1/ dif[mo,9];  k37 = k73;
        k84 = 1/ dif[mo,10]; k48 = k84;
        k95 = 1/ dif[mo,11]; k59 = k95;
        k106 = 1/dif[mo,12]; k610 = k106;
        k117 = 1/dif[mo,13]; k711 = k117;
        k89 = 1/ dif[mo,14]; k98 = k89;
        k910 = 1/dif[mo,15]; k109 = k910;
        k1011= 1/dif[mo,16]; k1110 = k1011;
        
        v01 = 1/ adv[mo,0]
        v12 = 1/ adv[mo,1]
        v23 = 1/ adv[mo,2]
        v45 = 1/ adv[mo,3]
        v56 = 1/ adv[mo,4]
        v67 = 1/ adv[mo,5]
        v40 = 1/ adv[mo,6]
        v51 = 1/ adv[mo,7]
        v62 = 1/ adv[mo,8]
        v73 = 1/ adv[mo,9]
        
        do_flow_adjust = True
        if do_flow_adjust:
            v73 = - v23
            v51 = v12 - v01
            v56 = v45 - v51
            v62 = v23 - v12
        
        wt = 5/3
        ws = 3/2
        
        data = [-k01-k04,k10,k40,k01,-k10-k12-k15,k21,k51,
                k12,-k21-k23-k26,k32,k62,k23,-k32-k37,k73,
                wt*k04,-wt*k40-k45-k48,k54,k84,wt*k15,k45,-k54-wt*k51-k59-k56,k65,k95,
                wt*k26,k56,-wt*k62-65-k67-k610,k76,k106,wt*k37,k67,-wt*k73-k76-k711,k117,
                ws*k48,-ws*k84-k89,k98,ws*k59,k89,-k98-k910-ws*k95,k109,
                ws*k610,k910,-k109-ws*k106-k1011,k1110,ws*k711,k1011,-ws*k117-k1110]
        row = [0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4,4,5,5,5,5,5,
               6,6,6,6,6,7,7,7,7,8,8,8,9,9,9,9,10,10,10,10,11,11,11]
        col = [0,1,4,0,1,2,5,1,2,3,6,2,3,7,0,4,5,8,1,4,5,6,9,
               2,5,6,7,10,3,6,7,11,4,8,9,5,8,9,10,6,9,10,11,7,10,11]
        D = bsr_matrix((data,(row,col)),shape=(12,12)).toarray()
        
        data = [v01-v40,v01,-v40,-v01,-v01+v12-v51,v12,-v51,
                -v12,-v12+v23-v62,v23,-v62,-v23,-v23-v73,-v73,
                v40,v40+v45,v45,v51,-v45,-v45+v51+v56,v56,
                v62,-v56,-v56+v62+v67,v67,v73,-v67,-v67+v73,0,0,0,0]
        row = [0,0,0,1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,5,6,6,6,6,7,7,7,8,9,10,11]
        col = [0,1,4,0,1,2,5,1,2,3,6,2,3,7,0,4,5,1,4,5,6,2,5,6,7,3,6,7,8,9,10,11]
        V = bsr_matrix((data,(row,col)),shape=(12,12)).toarray()
        
        W = spdiags([1,1,1,1,5/3,5/3,5/3,5/3,1,1,1,1],0,12,12).toarray()
        
        M = D+W@V/2; M = M*365
        N = np.size(M,1) - 1
        
        ohi = oh[mo,:]
        L_oh = Arr_A[i]*np.multiply(ohi,np.exp(-Arr_ER[i]/temp[mo,:]))/sec2year
        
        L_oh[8:11] = 1e-3
        if species == 'CFC11':
            L_oh[8:11] = 0.135
        elif species == 'CFC12':
            L_oh[8:11] = 0.054
        elif species == 'CFC113':
            L_oh[8:11] = 0.065
        elif species == 'CCl4':
            L_oh[8:11] = 0.172
        
        L = spdiags(L_oh[1:11],0,N,N).toarray()
        
        MII = M
        MII = np.delete(MII,0,0)
        MII = np.delete(MII,0,1)
        
        MIB = M[:,0]
        MIB = np.delete(MIB,0,0)
        
        I = eye(N).toarray()
        A = I-(MII-L)*dt*0.5
        B = I+(MII-L)*dt*0.5
        C = np.squeeze(B@XI0) + MIB*X1[t_idx]*dt
        lu, piv = lu_factor(A)
        XI1 = lu_solve((lu,piv),C)
        XI1[np.argwhere(XI1<0)] = 0
        XIvec = np.append(XIvec,XI1.reshape((1,N)),axis=0)
        XI0 = XI1
        