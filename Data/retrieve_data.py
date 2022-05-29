# +
'''
Contains methods that parses through directory of 600,000 stars to obtain the time
and magnitude data for the appropriate stars in directory.

This assumes that the data obtained from https://asas-sn.osu.edu/variables or 
https://drive.google.com/drive/folders/1IAtztpddDeh5XOiuxmLWdLUaT_quXkug is stored
in the local directory. This is a large file (~40 GB).
'''

import pandas as pd
import glob
import numpy as np
import pickle


# -

def obtain_irregular_data(df, all_files, num_stars=5000):
    '''
    Retrieves num_stars amount of irregular stars' magnitude data.
    
    Params:
        - df: the dataframe containing all of the irregular stars' metadata
        - all_files: list of all the files containing magnitude data
    '''
    irregular_data_limit = []
    irregular_names = df['asassn_name']

    count = 0
    for item in irregular_names:
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
        irregular_data_limit.append(data)
        count += 1
    
    # Save file
    with open('irregular_data_limit', 'wb') as f:
        pickle.dump(irregular_data_limit, f)

def obtain_other_data(df, all_files, save_file, num_stars=5000):
    '''
    Retreives num_stars amount of the other stars' magnitude and period data.
    
    Params:
        - df: the dataframe containing all of the periodic stars metadata
        - all_files: list of all the files containing magnitude data
        - save_file: the location to save the data to
    '''
    data_limit = []
    period_limit = []
    names = df['asassn_name']
    periods = df['period']
    count = 0

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

        period_limit.append(period)

        time_data = lines[:, 0]
        mag_data = lines[:, 1]
        data = np.stack((time_data, mag_data))
        data_limit.append(data)
        count += 1
    
    with open(save_file + '_periods_limit', 'wb') as f:
        pickle.dump(period_limit, f)
        
    with open(save_file + '_data_limit', 'wb') as f:
        pickle.dump(data_limit, f)

def load_dataframe(file_name):
    '''
    Loads the metadata into dataframes for each star type.
    
    Params:
        - file_name: the name of the file containing the medata for all stars
        
    Returns:
        - dataframes of the irregular, eb, mira, and rr stars
    '''
    df = pd.read_csv(file_name)
    df_irregular = df.loc[df['variable_type'].isin(['YSO', 'L', 'GCAS'])]
    df_eclipsing_binaries = df.loc[df['variable_type'].isin(['EB', 'EA', 'EW', 'ELL'])]
    df_short_period_rr_lyrae = df.loc[df['variable_type'].isin(['RRAB', 'RRC', 'RRD'])]
    df_long_period_mira_stars = df.loc[df['variable_type'].isin(['M'])]
    
    return df_irregular, df_eclipsing_binaries, df_short_period_rr_lyrae, df_long_period_mira_stars

if __name__ == '__main__':
    # Assume that metadata and raw data files are in ../Data/* directory.
    file = '../data/all_stars_meta.csv'
    df_irregular, df_eb, df_rr, df_mira = load_dataframe(file)
    all_files = glob.glob('../data/all_stars/*')
    
    # Extract and save time/magnitude data for irregulars (no period)
    obtain_irregular_data(df_irregular)
    
    # Extract and save time/magnitude data for eb, rr, mira (with periods)
    obtain_other_data(df_eb, 'eb')
    obtain_other_data(df_rr, 'rr')
    obtain_other_data(df_mira, 'mira')
