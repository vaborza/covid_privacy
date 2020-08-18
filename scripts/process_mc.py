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
    
    # Pull in pickle of bin sizes
    mc_data = pd.read_pickle(file_path)

    anon_df = pd.DataFrame()

    # Apply 5 desired operations on the pickle of bin sizes (ID'ing small bins, and getting median)
    anon_df['k1'] = mc_data.apply(lambda x: np.sum(x <= 1))
    anon_df['k5'] = mc_data.apply(lambda x: np.sum(x <= 5))
    anon_df['k10'] = mc_data.apply(lambda x: np.sum(x <= 10))
    anon_df['avg'] = mc_data.apply(lambda x: np.median(x))
    anon_df['std'] = mc_data.apply(lambda x: np.std(x))
    
    # Shape into an array of 3142 FIPS x 190 dates x 5 values of interest
    anon_arr = anon_df.values.reshape(3142,190,5)

    # Save
    with open('/data/victor/covid/anon_arr_'+str(time.time()).replace('.','-')+'.npy','wb') as f:
        np.save(f,anon_arr)

    print('Finished in '+str(time.time() - start_time) + ' seconds.')

if __name__ == '__main__':
    # Run on 25 workers, designed for 100 Monte Carlo simulations
    p = mp.Pool(25)
    p.map(run_mc_read,glob.glob('/data/victor/covid/*.pkl'))
