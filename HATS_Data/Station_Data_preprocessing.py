#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 18:40:41 2020

@author: hy337
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
from scipy import interpolate

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

def addressDuplicate(t,c):
    t_dup = []
    seen = []
    for i in t:
        if i not in seen:
            seen.append(i)
        else:
            if i not in t_dup:
                t_dup.append(i)
    for i in t_dup:
        dup_idx = np.argwhere(t==i)
        c[dup_idx[0]] = np.average(c[dup_idx])
        t = np.delete(t,dup_idx[1:],None)
        c = np.delete(c,dup_idx[1:],None)
    return t,c

work_dir = '/Users/hy337/Study/Data_Science/Projects/AGAGE_box_model/HATS_Data/'

species = ['sf6','cfc11','cfc12','cfc113','hcfc22','hcfc141b','hcfc142b','hfc134a','hfc152a','ccl4']
num_species = len(species)
filename = ['HATS_global_SF6','HATS_global_F11','HATS_global_F12','HATS_global_F113',
            'HCFC22_GCMS_flask','HCFC141B_GCMS_flask','HCFC142B_GCMS_flask',
            'hfc134a_GCMS_flask','hfc152a_GCMS_flask','HATS_global_CCl4']
file_name = {}
for s_idx in range(len(species)):
    s = species[s_idx]
    file_name[s] = filename[s_idx]
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
        
station = ['alt','sum','brw','mhd','thd','nwr','kum','mlo','smo','cgo','ush','psa','spo']
num_station = len(station)

for s in species:
    cct_full = {}
    cct_array = np.zeros((num_date,num_station))
    cct_array[:,:] = np.nan
    for i in range(num_date):
        cct_full[yymm[i]] = cct_array[i,:]
    if s in species[:4] or s in species[-1]:
        select_col = [0,1]
        select_col.extend(list(range(8,33,2)))
        cct = pd.read_csv(work_dir+file_name[s]+'.txt',delim_whitespace=True,
                          skiprows=84,usecols=select_col)
        timestamp = []
        for i in range(cct.shape[0]):
            date_slice = datetime.strptime(str(int(cct.iloc[i,0]*100+cct.iloc[i,1])),'%Y%m')
            timestamp.append(toYearFraction(date_slice))
        timestamp = np.array(timestamp)
        for station_idx in range(len(station)):
            timestamp_c = np.copy(timestamp)
            site_station = station[station_idx]
            concentr = cct.iloc[:,station_idx+2]
            concentr = concentr.to_numpy()
            nan_idx = np.argwhere(np.isnan(concentr))
            if len(nan_idx)>0:
                timestamp_c = np.delete(timestamp_c,nan_idx,None)
                concentr = np.delete(concentr,nan_idx,None)
            [timestamp_c,concentr]=addressDuplicate(timestamp_c,concentr)        
            f = interpolate.interp1d(timestamp_c,concentr,kind='cubic',bounds_error=False)
            concentr_new = np.ndarray.round(f(np.array(year_dec)),3)
            cct_array[:,station_idx] = concentr_new
        for i in range(num_date):
            cct_full[yymm[i]] = cct_array[i,:]
    else:
        cct = pd.read_csv(work_dir+file_name[s]+'.txt',sep='\t',skiprows=1)
        site = list(cct.site.unique())
        for site_station in site:
            if site_station in station:
                station_idx = int(np.argwhere(np.array(station)==site_station))
            col_tag = cct.columns[1]
            timestamp = cct.loc[cct['site']==site_station,col_tag]
            timestamp = timestamp.to_numpy()
            col_tag = cct.columns[5]
            concentr = cct.loc[cct['site']==site_station,col_tag]
            concentr = concentr.to_numpy()
            [timestamp,concentr]=addressDuplicate(timestamp,concentr)        
            f = interpolate.interp1d(timestamp,concentr,kind='cubic',bounds_error=False)
            concentr_new = f(np.array(year_dec))
            cct_array[:,station_idx] = concentr_new
        for i in range(num_date):
            cct_full[yymm[i]] = cct_array[i,:]
    cct_tab = pd.DataFrame.from_dict(cct_full,orient='index',columns=station)
    cct_tab.to_csv(work_dir+s+'_197701-202012.csv')
