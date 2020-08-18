# Victor Borza
# Aug 13, 2020
# Process the results of Monte Carlo sampling of cases from Johns Hopkins data against census data

import numpy as np
import pandas as pd
import time
import glob
import multiprocessing as mp

def run_mc_read(file_path):
    start_time = time.time()

#    sample_df = pd.read_pickle('../census_data/processed_19_data.pkl')
#    case_df = pd.read_csv('../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
#    case_df.drop(['UID','iso2','iso3','code3','Combined_Key','Admin2','Province_State','Country_Region',
#        'Lat','Long_'], axis=1, inplace=True)
#    case_df.set_index('FIPS',inplace=True)
#    case_df_rolled = case_df.rolling(7,axis=1,center=False).mean()
#    case_df = case_df[case_df.index.notnull()]
#    case_df.index = case_df.index.astype(int).map(str).str.zfill(5)
#    case_df = case_df[case_df.index.isin(sample_df['FIPS'])]
#    case_df.sort_index()
#    sample_df.sort_values(by=['FIPS'])

    mc_data = pd.read_pickle(file_path)

    anon_df = pd.DataFrame()

    anon_df['k1'] = mc_data.apply(lambda x: np.sum(x <= 1))
    anon_df['k5'] = mc_data.apply(lambda x: np.sum(x <= 5))
    anon_df['k10'] = mc_data.apply(lambda x: np.sum(x <= 10))
    anon_df['avg'] = mc_data.apply(lambda x: np.mean(x))
    anon_df['std'] = mc_data.apply(lambda x: np.std(x))

    anon_arr = anon_df.values.reshape(3142,190,5)

    with open('/data/victor/covid/anon_arr_'+str(time.time()).replace('.','-')+'.npy','wb') as f:
        np.save(f,anon_arr)

    print('Finished in '+str(time.time() - start_time) + ' seconds.')

if __name__ == '__main__':
    p = mp.Pool(24)
    p.map(run_mc_read,glob.glob('/data/victor/covid/*.pkl'))
