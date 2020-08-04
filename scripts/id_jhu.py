# Victor Borza
# Jul 30, 2020
# Read the Johns-Hopkins dataset and evaluate for reidentification risk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import seaborn as sns
import random
import plotly.figure_factory as ff
import plotly.express as px

# Apply the Golle formula, log must be used to prevent overflows
def estimate_anon(pop,bins,k_level):
    i = k_level
    return sum(np.exp(np.log(comb(pop,i,exact=False)) + \
    np.log(bins) * (1-pop) + \
    np.log((bins-1)) * (pop-i)) for i in range(1,k_level+1))

###

def gen_risk_ratio(sex_bins,race_bins,ethnicity_bins,age_specificity,age_cap,k_level):
    
    # Generate heatmap over time for 50 randomly selected FIPS codes
    # sex_bins: Bins by sex
    # race_bins: Bins by race 
    # ethnicity_bins: Bins by ethnicity
    # age_specificity: In years, with 1 being defined to the year
    # age_cap: Maximum age before we group all together
    # k_level: Set level of k-anonymity desired 

    df = pd.read_csv('../csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
    df.drop(['UID','iso2','iso3','code3','Combined_Key','Admin2','Province_State','Country_Region',
        'Lat','Long_'], axis=1,inplace=True)

    df.set_index('FIPS',inplace=True)

    # Calculate the rolling average using a 7-day window
    df_rolled = df.rolling(7,axis=1,center=False).mean()

    # Generate age bins from specified age values, then calculate total number of bins
    age_bins = np.rint(age_cap / age_specificity) + 1 #Round to integer, then add 1 for the 'and up' case
    total_bins = sex_bins * race_bins * ethnicity_bins * age_bins

    # Generate estimate of risk as a proportion of individuals with desired k anonymity or less
    df_at_risk = estimate_anon(df_rolled,total_bins,k_level)

    # Remove leading 0's generated by the rolling average function
    df_at_risk.replace(0,np.nan,inplace=True)
    return df_at_risk / df_rolled

###

def gen_heatmap(seed=7312020,sex_bins=2,race_bins=7,ethnicity_bins=2,
        age_specificity=1,age_cap=90,k_level=1):
    
    # Generate heatmap over time for 50 randomly selected FIPS codes, specify seed if desired
    # Inputs map to gen_risk_ratio, defaults are based on OMB data

    random.seed(seed)

    df_risk_ratio = gen_risk_ratio(sex_bins,race_bins,ethnicity_bins,age_specificity,age_cap,k_level)
    dates = pd.to_datetime(df_risk_ratio.columns.values)
    loc_pool = random.sample(range(len(df_risk_ratio)),50)

    ax=sns.heatmap(df_risk_ratio.iloc[loc_pool].to_numpy(),cmap='jet')
    plt.show()

###

def threshold_re_id(sex_bins=2,race_bins=7,ethnicity_bins=2,age_specificity=1,
        age_cap=90,k_level=1,id_cutoff=0.05):

    # For a given sharing profile, estimate the number of counties that can safely share that data
    # using a desired k level and population percentage
    # id_cutoff: The percentage of the population that is acceptable to be identifiable at the given
    #            k level or less

    df_risk_ratio = gen_risk_ratio(sex_bins,race_bins,ethnicity_bins,age_specificity,age_cap,k_level)
    
    # Identify counties where risk is less than the determined cutoff
    safe_locales = (df_risk_ratio < id_cutoff).to_numpy().sum(axis=0) / len(df_risk_ratio.index)
    
    fig, ax = plt.subplots()
    plt.plot(df_risk_ratio.columns.values,safe_locales)
    
    # Label by week
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 7 != 0:
            label.set_visible(False)
    plt.xticks(rotation=60)
    plt.title('Proportion of counties with an expected re-identification risk of ' + str(id_cutoff) +
            ' at k-anonymity of ' + str(k_level))
    ax.annotate('# Sexes: ' + str(sex_bins) + '\n# Races: ' + str(race_bins) +
            '\n# Ethnicities: ' + str(ethnicity_bins) + '\nAge Specificity: ' +
            str(age_specificity) + '\nAge Cap: ' + str(age_cap),xy=(0.2,0.7),xycoords='figure fraction')
    plt.show()

###

def gen_geomap(sex_bins=2,race_bins=7,ethnicity_bins=2,age_specificity=1,age_cap=90,k_level=1,
        date='7/29/20'):

    # Show COVID re-identification risk for the US (as a choropleth map) on a given date
    # date: string input (no leading 0's) of the desired date
    
    df_risk_ratio = gen_risk_ratio(sex_bins,race_bins,ethnicity_bins,age_specificity,age_cap,k_level)
    
    # Strip out all values of NaN either in the FIPS code or COVID cases, that way it shows up white
    df_risk_ratio = df_risk_ratio[df_risk_ratio.index.notna() & df_risk_ratio[date].notna()]

    fips = (df_risk_ratio.index.values)
    values = df_risk_ratio[date]
    
    # N.B. Plasma is perceptually linear but the chosen endpoints are not, as data are more useful
    # at the extremes
    cmap = px.colors.sequential.Plasma
    endpts = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    fig = ff.create_choropleth(fips=fips,values=values,colorscale=cmap,round_legend_values=False,
            title=('COVID Data Sharing Risk on ' + date + ' at k-level ' + str(k_level) +
            '<br />Using: ' + str(sex_bins) + ' sexes, ' + str(race_bins) + ' races, ' + 
            str(ethnicity_bins) + ' ethnicities, Age in ' + str(age_specificity) + 
            '-year bins, until ' + str(age_cap) + ' years'), 
            binning_endpoints = endpts, legend_title='Re-ID risk')
    fig.layout.template = None
    fig.show()

###
