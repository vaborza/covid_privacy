# Victor Borza
# Aug 6, 2020
# Estimate the risk surface in Florida's open data sharing by linking against 2019 data

import pandas as pd
import numpy as np

def re_id(k_level=1):
    fl_df = pd.read_pickle('../fl_data/FL_line_data.pkl')
    census_df = pd.read_pickle('../census_data/census_usa_2019.pkl')
    census_df = census_df[census_df['STNAME'] == 'Florida']

    census_df['CTYNAME'] = census_df['CTYNAME'].str.replace(r' County$','')

    print(census_df['CTYNAME'])

