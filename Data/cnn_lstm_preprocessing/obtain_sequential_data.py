import pandas as pd
import numpy as np
from PyAstronomy.pyasl import foldAt
from astropy.timeseries import LombScargle
import pickle

def get_padded_sequence_data(x_data, save_file, max_length = 500):
    sequence_data = []
    
    for item in x_data:
        item = np.array(item).astype(float)
    
        sorted_sequence = item[1][np.argsort(item[0])]
        
        if (len(sorted_sequence) > max_length):
            sequence_data.append(sorted_sequence[0:max_length])
        elif (len(sorted_sequence) < max_length):
            sequence_data.append(np.pad(sorted_sequence, (0, max_length - len(sorted_sequence))))
        else:
            sequence_data.append(sorted_sequence)
    
    sequence_data = np.array(sequence_data)
    with open(save_file, 'rb') as f:
        pickle.dump(sequence_data, f)

def get_phase_folded_data_irregular(irregular_data):
    irregular_data_phase_folded = []
    
    for item in irregular_data:
        # Sort time data and magnitude data
        sorted_time_data = item[0][np.argsort(item[0])]
        sorted_mag_data = item[1][np.argsort(item[1])]

        # Get power in frequency space
        frequency, power = LombScargle(sorted_time_data, sorted_mag_data).autopower(minimum_frequency=0.0005,
                                                       maximum_frequency=50)
        # Use frequency with max power to get "best period"
        best_frequency = frequency[np.argmax(power)]
        period = 1 / best_frequency

        #Phase fold data
        T_0 = sorted_time_data[np.argmin(sorted_mag_data)]
        phases = foldAt(sorted_time_data, period, T0=T_0)

        sorted_phases = phases[np.argsort(phases)]
        phase_sorted_mag = sorted_mag_data[np.argsort(phases)]
        data = np.stack((sorted_phases, phase_sorted_mag))

        irregular_data_phase_folded.append(data)
    
    return irregular_data_phase_folded

def periodic_phase_folded_data(mag_data, period_data):
    data_phase_folded = []
    for item, period in zip(mag_data, period_data):
        time_data = item[0]
        mag_data = item[1]
        
        T_0 = time_data[np.argmin(mag_data)]
        phases = foldAt(time_data, period, T0=T_0)
        sorted_phases = phases[np.argsort(phases)]
        phase_sorted_mag = mag_data[np.argsort(phases)]
        data = np.stack((sorted_phases, phase_sorted_mag))
        data_phase_folded.append(data)  
    
    return data_phase_folded

def process_phase_folded_data(phase_folded_data, save_file, max_length = 500):
    padded_phase_folded_data = []
    
    for item in phase_folded_data:
        phases = item[0]
        mags = item[1]

        if (len(phases) > max_length):
            padded_phase_folded_data.append([phases[0:max_length], mags[0:max_length]])
        elif (len(item[0]) < max_length):
            padded_phase_folded_data.append([np.pad(phases, (0, max_length - len(phases))),
                                            np.pad(mags, (0, max_length - len(mags)))])
        else:
            padded_phase_folded_data.append([phases, mags])
    
    diff_phase_folded = []
    
    # Take successive differences in data for normalization
    for item in padded_phase_folded_data:
        phases = item[0]
        mags = item[1]

        diff_phases = [t - s for s, t in zip(phases, phases[1:])]
        diff_mags = [t - s for s, t in zip(mags, mags[1:])]

        diff_phase_folded.append(np.stack((diff_phases, diff_mags)))

    diff_phase_folded = np.array(diff_phase_folded)
    
    with open(save_file, 'wb') as f:
        pickle.dump(diff_phase_folded_data)

if __name__ == '__main__':
    irreg_data_file = '../raw_magnitude_data/irregular_data_limit'
    eb_data_file = '../raw_magnitude_data/eb_data_limit'
    eb_periods_file = '../period_data/eb_period_limit'
    mira_data_file = '../raw_magnitude_data/mira_data_limit'
    mira_periods_file = '../period_data/mira_period_limit'
    rr_data_file = '../raw_magnitude_data/rr_data_limit'
    rr_periods_file = '../period_data/rr_period_limit'
    
    # Open necessary files to get raw data
    with open(irreg_data_file, 'rb') as f:
        irregular_data = pickle.load(f)

    with open(eb_data_file, 'rb') as f:
        eb_data = pickle.load(f)

    with open(mira_data_file, 'rb') as f:
        mira_data = pickle.load(f)

    with open(rr_data_file, 'rb') as f:
        rr_data = pickle.load(f)
        
    with open(eb_periods_file, 'rb') as f:
        eb_periods = pickle.load(f)
    
    with open(mira_periods_file, 'rb') as f:
        mira_periods = pickle.load(f)

    with open(rr_periods_file, 'rb') as f:
        rr_periods = pickle.load(f)
    
    x_data = irregular_data + eb_data + mira_data + rr_data
    # Get zero-padded sequential array of magnitudes sorted in time domain
    get_padded_sequence_data(x_data, 'padded_sequence_data')
    
    # Get phase folded data
    irregular_phase_folded = get_phase_folded_data_irregular(irregular_data)
    eb_phase_folded = periodic_phase_folded_data(eb_data, eb_periods)
    mira_phase_folded = periodic_phase_folded_data(mira_data, mira_periods)
    rr_phase_folded = periodic_phase_folded_data(rr_data, rr_periods)
    
    # Get differenced/normalized phase data
    all_phase_folded = irregular_phase_folded + eb_phase_folded + mira_phase_folded + rr_phase_folded
    process_phase_folded_data(all_phase_folded, 'diff_phase_folded_data')