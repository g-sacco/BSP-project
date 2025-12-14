import numpy as np

def max_min_difference(signal):
    '''Calculate the difference between the maximum and minimum value in a vector'''
    max = np.max(signal)
    min = np.min(signal)
    delta = max - min
    return delta

def is_zero_crossing(signal, i):
    '''
    Calculate if two point are a zero-crossing. 
    If they are return the AMMp (Adjacent Modulus Maxima pair) value, otherwise return 0.
    '''
    # Check if there is a zero crossing between i-1 and i
    if (signal[i-1] <= 0 and signal[i] > 0) or (signal[i-1] > 0 and signal[i] <= 0):
        # search left
        j = i - 1
        loc_max = abs(signal[j])
        while j > 0 and abs(signal[j-1]) >= loc_max:
            j -= 1
            loc_max = abs(signal[j])
        left_peak = abs(signal[j])
        left_peak_pos = j

        # search right
        j = i
        loc_max = abs(signal[j])
        while  j < len(signal)-1 and abs(signal[j+1]) >= loc_max:
            j += 1
            loc_max = abs(signal[j])
        right_peak = abs(signal[j])
        right_peak_pos = j
        # calculate Adjacent Modulus Maxima pair
        AMMp = left_peak + right_peak
    else:
        AMMp = 0
        left_peak_pos = None
        right_peak_pos = None
    return AMMp, left_peak_pos, right_peak_pos

def qrs_detection(details, fs):
    '''
    QRS detection algorithm (R peak detection) based on the detail coefficients at levels 2 and 3 (d^2 and d^3).

    Parameters:
    -----------
    details : dict
        Dictionary of detail coefficients at each scale k (key = k, value = d^k
    fs : int
        Sampling frequency of the ECG signal
    
    Returns:
    --------
    r_peaks : array
        Array of detected R peak positions (indices in the signal)
    qrs_d2_peaks_pos : list of tuples
        List of tuples containing the left and right AMMp positions for each detected QRS
    '''
    # detail levels used
    d2=details[2]
    d3=details[3]
    r_peaks = []                  # found qrs postions
    qrs_d2_delta = []                   # qrs min-max delta on d^2
    qrs_d3_delta = []                   # qrs min-max delta on d^3
    qrs_d2_peaks_pos = []               # qrs peak positions on d^2
    qrs_found = 0                       # number of QRS found
    deltaWQRS = np.right_shift(fs, 2)   # observation window = 250 ms
    deltaW100 = int(fs * 0.1)           # half-window samples number for detection rules
    i_zc = int(1)                       # index for zero-crossing search
    signal_length = len(d2)

     # search zero-crossings in detail level 2^2
    while i_zc < signal_length:
        AMMp, left_peak_pos, right_peak_pos = is_zero_crossing(d2, i_zc)   # Adjacent Modulus Maxima pair
        # if zero-crossing (ZCs) is found (AMMp>0) start looking for 
        # better ZCs (bigger AMMp) in the observation window
        if AMMp > 0:
            j_zc = i_zc + 1
            # iterate in the observation window looking for better QRS-candidates
            while j_zc < i_zc + deltaWQRS and j_zc < signal_length:
                AMMp_new, left_peak_pos_new, right_peak_pos_new = is_zero_crossing(d2, j_zc)
                # if a bigger AMMp is found, restart the obs. window
                if AMMp_new > AMMp:
                    AMMp = AMMp_new
                    left_peak_pos = left_peak_pos_new
                    right_peak_pos = right_peak_pos_new
                    i_zc = j_zc # new QRS candidate found
                else:
                    j_zc += 1

            # QRS candidate decision  
            delta_window = max(0, i_zc-deltaW100), min(signal_length, i_zc+deltaW100+1)  
            e_start = max(0, qrs_found-4)
            zc_d2_delta = max_min_difference(d2[delta_window[0]:delta_window[1]])
            if len(qrs_d2_delta[e_start:qrs_found]) > 0:
                # compute (11 * sum) / (32 * len) 
                sum_d2 = sum(qrs_d2_delta[e_start:qrs_found])
                len_d2 = len(qrs_d2_delta[e_start:qrs_found])
                if len_d2 == 4:
                    epsilon_d2 = (11 * sum_d2) >> 7
                else:
                    epsilon_d2 = (11 * sum_d2) // (32 * len_d2)  
            else:
                epsilon_d2 = -np.inf
            # first test on d^2
            if zc_d2_delta > epsilon_d2:
                zc_d3_delta = max_min_difference(d3[delta_window[0]:delta_window[1]])
                if len(qrs_d3_delta[e_start:qrs_found]) > 0:
                    # compute (11 * sum) / (32 * len) 
                    sum_d3 = sum(qrs_d3_delta[e_start:qrs_found])
                    len_d3 = len(qrs_d3_delta[e_start:qrs_found])
                    if len_d3 == 4:
                        epsilon_d3 = (11 * sum_d3) >> 7
                    else:   
                        epsilon_d3 = (11 * sum_d3) // (32 * len_d3)  
                else:
                    epsilon_d3 = -np.inf
                # second test on d^3
                if zc_d3_delta > epsilon_d3:
                    # QRS-candidate ACCEPTED
                    r_peaks.append(np.int32(i_zc))
                    qrs_d2_delta.append(zc_d2_delta)
                    qrs_d3_delta.append(zc_d3_delta)
                    qrs_d2_peaks_pos.append((left_peak_pos, right_peak_pos))
                    qrs_found += 1
            i_zc += deltaWQRS + 1
        else:
            i_zc += 1

    return r_peaks, qrs_d2_peaks_pos
