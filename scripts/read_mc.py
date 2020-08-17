# Victor Borza
# Aug 13, 2020
# Process the results of Monte Carlo sampling of cases from Johns Hopkins data against census data

import numpy as np
import pandas as pd
import time
import glob
import multiprocessing as mp
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

def plot_k_timelines():
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

    stats_arr = np.load('../data/stats_arr.npy')

    stats_arr_norm = np.divide(stats_arr,case_arr.reshape(1,3142,190,1))

    # Get the average bin size by day
    bin_mean = np.nanmedian(stats_arr[:,:,:,3],axis=(0,1))
    bin_std = np.nanmean(stats_arr[:,:,:,4],axis=(0,1))

    # Get the proportion of individuals that are in a bin of 1
    bin_k1 = np.nanmean(stats_arr_norm[:,:,:,0],axis=(0,1))
    bin_k5 = np.nanmean(stats_arr_norm[:,:,:,1],axis=(0,1))
    bin_k10 = np.nanmean(stats_arr_norm[:,:,:,2],axis=(0,1))

    #plt.plot(case_df.columns.values,bin_mean)
    fig,ax = plt.subplots()

    plt.errorbar(case_df.columns.values,bin_mean,bin_std)
    #plt.plot(case_df.columns.values,bin_k1,label='k = 1')
    #plt.plot(case_df.columns.values,bin_k5,label='k = 5')
    #plt.plot(case_df.columns.values,bin_k10,label='k = 10')
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 7 != 0:
            label.set_visible(False)
    plt.xticks(rotation=60)
    plt.title('Median (across counties) bin size (k) over time')
    plt.legend()

    plt.show()

    print('Finished in '+str(time.time() - start_time) + ' seconds.')

def gen_geomap(date='7/29/20',metric='k1'):

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

    stats_arr = np.load('../data/stats_arr.npy')

    stats_arr_norm = np.divide(stats_arr,case_arr.reshape(1,3142,190,1))

    # Get the average bin size by day
    bin_median = np.nanmedian(stats_arr[:,:,:,3],axis=(0))
    bin_std = np.nanmean(stats_arr[:,:,:,4],axis=(0))

    # Get the proportion of individuals that are in a bin of 1
    bin_k1 = np.nanmean(stats_arr_norm[:,:,:,0],axis=(0))
    bin_k5 = np.nanmean(stats_arr_norm[:,:,:,1],axis=(0))
    bin_k10 = np.nanmean(stats_arr_norm[:,:,:,2],axis=(0))

    fips = (case_df.index.values)
    if metric == 'k1': values = bin_k1[:,case_df.columns.get_loc(date)] * 100
    elif metric == 'k5': values = bin_k5[:,case_df.columns.get_loc(date)]  * 100
    elif metric == 'k10': values = bin_k10[:,case_df.columns.get_loc(date)] * 100
    elif metric == 'median': values = bin_median[:,case_df.columns.get_loc(date)]
    else: 
        print('Metric invalid')
        return(0)

    #print(bin_k1.shape)
    #print(case_df.columns.get_loc('7/29/20'))

    # N.B. Plasma is perceptually linear but the chosen endpoints are not, as data are more useful
    # at the extremes
    
    values = np.nan_to_num(values)

    cmap = px.colors.sequential.Plasma
    if metric == 'median':
        endpts = [0.01, 25, 100, 250, 1000, 5000, 10000, 50000]
    else:
        endpts = [0.01, 0.1, 0.5, 1, 5, 10, 25, 50]
    
    fig = ff.create_choropleth(fips=fips,values=values,colorscale=cmap,round_legend_values=False,
            title=('COVID Data Sharing Risk on ' + date + ' showing ' + str(metric)),
            binning_endpoints = endpts, legend_title='Median Bin Size', asp=2.9,
            centroid_marker={'opacity':0})
    fig.layout.template = None
    fig.show()    
