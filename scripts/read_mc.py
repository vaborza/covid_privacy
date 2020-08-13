# Victor Borza
# Aug 13, 2020
# Process the results of Monte Carlo sampling of cases from Johns Hopkins data against census data

import numpy as np
import pandas as pd
import time
import glob
import multiprocessing as mp
import matplotlib.pyplot as plt


start_time = time.time()

# Initialize the case array to get case counts
sample_df = pd.read_pickle('../census_data/processed_19_data.pkl')
case_df = pd.read_csv('../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
case_df.drop(['UID','iso2','iso3','code3','Combined_Key','Admin2','Province_State','Country_Region',
    'Lat','Long_'], axis=1, inplace=True)
case_df.set_index('FIPS',inplace=True)
case_df_rolled = case_df.rolling(7,axis=1,center=False).mean()
case_df = case_df[case_df.index.notnull()]
case_df.index = case_df.index.astype(int).map(str).str.zfill(5)
case_df = case_df[case_df.index.isin(sample_df['FIPS'])]
case_df.sort_index()
sample_df.sort_values(by=['FIPS'])
case_arr = case_df.to_numpy()

# Initialize the list to capture all 100 Monte Carlo samples
stats_list = []
data_path = glob.glob('/data/victor/covid/*.npy')

# Read in the Monte Carlo samples and convert to an array
# Array size: 100 samples x 3142 locations x 190 dates x 5 stats (k1,k5,k10,mean,std)
for mc_sample in data_path:
    stats_list.append(np.load(mc_sample))

stats_arr = np.array(stats_list)

with open('../data/stats_arr.npy','wb') as f:
    np.save(f,stats_arr)

#stats_arr_norm = np.divide(stats_arr,case_arr.reshape(1,3142,190,1))

# Get the average bin size by day
#bin_mean = np.nanmean(stats_arr[:,:,:,3],axis=(0,1))
#bin_std = np.nanmean(stats_arr[:,:,:,4],axis=(0,1))

# Get the proportion of individuals that are in a bin of 1
#bin_k1 = np.nanmean(stats_arr_norm[:,:,:,0],axis=(0,1))
#bin_k5 = np.nanmean(stats_arr_norm[:,:,:,1],axis=(0,1))
#bin_k10 = np.nanmean(stats_arr_norm[:,:,:,2],axis=(0,1))

#plt.plot(case_df.columns.values,bin_mean)
#plt.errorbar(case_df.columns.values,bin_mean,bin_std)
#plt.show()

#print('Finished in '+str(time.time() - start_time) + ' seconds.')

