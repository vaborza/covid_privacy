# Simply read the TN demographic data
# Victor Borza
# Jul 28, 2020

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r"../data/TN_race_eth_sex.xlsx")

df_white = df[df['CAT_DETAIL'].str.contains("White")]
df_black = df[df['CAT_DETAIL'].str.contains("Black")]
df_pending = df[df['CAT_DETAIL'].str.contains("Pending") & df['Category'].str.contains("RACE")]

pct_white = df_white['Cat_Percent'].to_numpy()
pct_black = df_black['Cat_Percent'].to_numpy()
pct_pending = df_pending['Cat_Percent'].to_numpy()
total_pct = pct_white + pct_black + pct_pending

dates = df_white['Date'].to_numpy()

fig, ax = plt.subplots()
ax.plot(dates,pct_white,label='White')
ax.plot(dates,pct_black,label='Black')
ax.plot(dates,total_pct,label='Total B + W + pending')
ax.plot(dates,pct_pending,label='Pending')
ax.legend()
plt.show()
