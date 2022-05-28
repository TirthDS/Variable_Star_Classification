import pandas as pd
import pickle
import glob
import numpy as np

file = 'cs_229_dat/asassn_catalog_full.csv'
df = pd.read_csv(file)
df_irregular = df.loc[df['variable_type'].isin(['YSO', 'L', 'GCAS'])]
df_eclipsing_binaries = df.loc[df['variable_type'].isin(['EB', 'EA', 'EW', 'ELL'])]
df_short_period_rr_lyrae = df.loc[df['variable_type'].isin(['RRAB', 'RRC', 'RRD'])]
df_long_period_mira_stars = df.loc[df['variable_type'].isin(['M'])]

all_files = glob.glob('cs_229_dat/vardb_files/*') # Gives list of all files containing actual data

# Function to parse only the irregular data
def parse_irregular_data(names, limit):
    count = 0
    for item in names:
        if (count >= 5000):
            break
        
        name = item.replace(" ", "")
        data_file = [file for file in all_files if file[23:-4]==name][0]
        f = open(data_file, "r")
        lines = f.readlines()[2:]
        lines = np.array([line.split() for line in lines]).astype(float)
        if (len(lines) == 0):
            continue
        
        time_data = lines[:, 0]
        mag_data = lines[:, 1]
        data = np.stack((time_data, mag_data))
        limit.append(data)
        
        count += 1

# Function to parse only the eb, mira, and the rr lyrae
def parse_other_data(names, periods):
    for item, period in zip(names, periods):
        if (count >= 5000):
            break
        
        name = item.replace(" ", "")
        data_file = [file for file in all_files if file[23:-4]==name][0]
        f = open(data_file, "r")
        lines = f.readlines()[2:]
        lines = np.array([line.split() for line in lines]).astype(float)
        if (len(lines) == 0):
            continue
            
        rr_period_limit.append(period)
        
        time_data = lines[:, 0]
        mag_data = lines[:, 1]
        data = np.stack((time_data, mag_data))
        rr_data_limit.append(data)
        
        count += 1


irregular_data_limit = []
irregular_names = df_irregular['asassn_name']
parse_irregular_data(irregular_names, irregular_data_limit)

eb_data_limit = []
eb_period_limit = []
eb_names = df_eclipsing_binaries['asassn_name']
eb_periods = df_eclipsing_binaries['period']
count = 0
parse_other_data(eb_names, eb_periods)

mira_data_limit = []
mira_period_limit = []
mira_names = df_eclipsing_binaries['asassn_name']
mira_periods = df_eclipsing_binaries['period']
count = 0
parse_other_data(mira_names, mira_periods)

rr_data_limit = []
rr_period_limit = []
rr_names = df_short_period_rr_lyrae['asassn_name']
rr_periods = df_short_period_rr_lyrae['period']
count = 0
parse_other_data(rr_names, rr_periods)