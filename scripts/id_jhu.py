# Victor Borza
# Jul 30, 2020
# Read the Johns-Hopkins dataset and evaluate for reidentification risk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import seaborn as sns
import random

# Apply the Golle formula, log must be used to prevent overflows
def estimate_anon(pop,bins,k_level):
    i = k_level
    return sum(np.exp(np.log(comb(pop,i,exact=False)) + \
    np.log(bins) * (1-pop) + \
    np.log((bins-1)) * (pop-i)) for i in range(1,k_level+1))

###

def gen_risk_ratio(seed,sex_bins,race_bins,age_specificity,age_cap,k_level):
    
    # Generate heatmap over time for 50 randomly selected FIPS codes
    # sex_bins: Bins by sex
    # race_bins: Bins by race 
    # age_specificity: In years, with 1 being defined to the year
    # age_cap: Maximum age before we group all together
    # k_level: Set level of k-anonymity desired
    
    random.seed(seed) 

    df = pd.read_csv('../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    df.drop(['UID','iso2','iso3','code3','FIPS','Admin2','Province_State','Country_Region','Lat','Long_'],
            axis=1,inplace=True)
    df.set_index('Combined_Key',inplace=True)

    # Generate age bins from specified age values, then calculate total number of bins
    age_bins = np.rint(age_cap / age_specificity) + 1 #Round to integer, then add 1 for the 'and up' case
    total_bins = sex_bins * race_bins * age_bins

    # Generate estimate of risk as a proportion of individuals with desired k anonymity or less
    df_at_risk = estimate_anon(df,total_bins,k_level)
    return df_at_risk / df

###

def gen_heatmap(seed=7312020,sex_bins=2,race_bins=5,age_specificity=1,age_cap=90,k_level=1):
    
    # Generate heatmap over time for 50 randomly selected FIPS codes, specify seed if desired
    # Inputs map to gen_risk_ratio, defaults are based on OMB data

    df_risk_ratio = gen_risk_ratio(seed,sex_bins,race_bins,age_specificity,age_cap,k_level)
    dates = pd.to_datetime(df_risk_ratio.columns.values)
    loc_pool = random.sample(range(len(df_risk_ratio)),50)

    ax=sns.heatmap(df_risk_ratio.iloc[loc_pool].to_numpy(),cmap='jet')
    plt.show()

