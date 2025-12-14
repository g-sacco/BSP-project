import numpy as np

def is_local_max(signal, i):
    """Check if position i is a local maximum in absolute value"""
    if i <= 0 or i >= len(signal) - 1:
        return False
    val = abs(signal[i])
    return val > abs(signal[i-1]) and val >= abs(signal[i+1])

def qrs_delineation(details, fs, r_peaks, qrs_d2_peaks_pos):
    """
    QRS delineation at scale 2^2 following Di Marco & Chiari (2011)
    
    Parameters:
    -----------
    details : dict
        Dictionary of wavelet detail coefficients at different scales
    fs : int
        Sampling frequency (samples/s)
    r_peaks : list
        R-peak positions (zero-crossings)
    qrs_d2_peaks_pos : list of tuples
        (npre, npost) positions for each QRS
    
    Returns:
    --------
    QRS_onsets : list
        List of QRS onset positions
    QRS_offsets : list
        List of QRS offset positions
    """

    d2 = details[2]
    fixed_window = int(fs * 0.120)  # 120 ms fixed window in samples
    QRS_onsets = []
    QRS_offsets = []

    for key, qrs_pos in enumerate(r_peaks):
        npre = qrs_d2_peaks_pos[key][0]
        npost = qrs_d2_peaks_pos[key][1]

        # Use absolute values for threshold calculations (not specified in paper)
        d2_npre_abs = abs(int(d2[npre]))
        d2_npost_abs = abs(int(d2[npost]))
        
        # ===== QRS ONSET SEARCH =====
        # Thresholds from equation (5.6)
        e1_onset = (d2_npre_abs + d2_npost_abs) >> 5  # divide by 32
        e2_onset = (d2_npre_abs + d2_npost_abs) >> 4  # divide by 16
        
        onset = None
        candidate_onset = None  # Store first e1 crossing
        n_left = None  # Left-most modulus maximum
        
        window_start = max(0, npre - fixed_window)
        i = npre
        
        # Search backwards from npre
        while i > window_start:
            mod_d2i = abs(d2[i])
            
            # Store first crossing of e1_onset (will be used if no maxima found)
            if candidate_onset is None and mod_d2i <= e1_onset:
                candidate_onset = i
            
            # Check for local maximum
            if is_local_max(d2, i):
                if mod_d2i <= e2_onset:
                    # Found maximum below e2_onset threshold (NOISE) - stop here
                    break
                else:
                    # Found maximum above threshold - keep searching but store it
                    n_left = i
            
            i -= 1
        
        # Determine onset based on what was found
        if n_left is None:
            # No modulus maxima found - use first e1 crossing or window edge
            onset = candidate_onset if candidate_onset is not None else window_start
        else:
            # Modulus maxima found - search from n_left for e3 threshold
            e3_onset = abs(int(d2[n_left])) >> 2  # divide by 4
            
            i = n_left
            while i > window_start:
                if abs(d2[i]) <= e3_onset:
                    onset = i
                    break
                i -= 1
            
            if onset is None:
                onset = window_start
        
        QRS_onsets.append(int(onset))
        
        # ===== QRS OFFSET SEARCH =====
        # Thresholds from equation (5.8)
        e1_offset = d2_npost_abs >> 2  # divide by 4
        e2_offset = d2_npost_abs >> 3  # divide by 8
        
        offset = None
        candidate_offset = None
        n_right = None
        
        window_end = min(len(d2) - 1, npost + fixed_window)
        i = npost
        
        # Search forwards from npost
        while i < window_end:
            mod_d2i = abs(d2[i])
            
            # Store first crossing of e1_offset
            if candidate_offset is None and mod_d2i <= e1_offset:
                candidate_offset = i
            
            # Check for local maximum
            if is_local_max(d2, i):
                if mod_d2i <= e2_offset:
                    # Found maximum below e2_offset threshold (NOISE) - stop here
                    break
                else:
                    n_right = i
            
            i += 1
        
        # Determine offset based on what was found
        if n_right is None:
            offset = candidate_offset if candidate_offset is not None else window_end
        else:
            # Search from n_right for e3 threshold
            e3_offset = abs(int(d2[n_right])) >> 2  # divide by 4
            
            i = n_right
            while i < window_end:
                if abs(d2[i]) <= e3_offset:
                    offset = i
                    break
                i += 1
            
            if offset is None:
                offset = window_end
        
        QRS_offsets.append(int(offset))

    return QRS_onsets, QRS_offsets