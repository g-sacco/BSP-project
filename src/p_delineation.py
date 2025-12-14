import numpy as np
from qrs_detection import is_zero_crossing

def interbeat_interval(r_peaks, pos):
    """return half the samples distance between the actual R peak and the precedent"""
    if pos > 1 and pos < len(r_peaks):
        return np.right_shift(r_peaks[pos] - r_peaks[pos-1], 1)
    else:
        return np.inf
    
def biphasic_first_condition(d3, n_pre, n_post):
    """application of the (6.1) condition (from the paper) for a p_peak to be bi-phasic"""
    d3_n_post_mul2 = 2 * abs(d3[n_post])
    d3_n_post_div2 = np.right_shift(abs(d3[n_post]), 1)
    d3_n_pre = abs(d3[n_pre])

    if d3_n_post_div2 <= d3_n_pre and d3_n_pre <= d3_n_post_mul2:
        return True
    else:
        return False
    
def biphasic_second_condition(AMMp_P, AMMp_bi):
    """application of the (6.2) condition (from the paper) for a p_peak to be bi-phasic"""
    if AMMp_bi > np.right_shift(3 * AMMp_P, 2):
        return True
    else:
        return False


def p_delineation(details, fs, QRS_onset, r_peaks):
    """
    P wave delineation following Di Marco & Chiari (2011)

    Parameters:
    -----------
    details : dict
        Dictionary of wavelet detail coefficients at different scales
    fs : int
        Sampling frequency (samples/s)
    QRS_onset : list
        List of QRS onset positions
    r_peaks : list
        R-peak positions 

    Returns:
    --------
    p_peaks : list
        List of P peak positions
    p_onsets : list
        List of P onset positions
    p_offsets : list
        List of P offset positions
    """

    d3 = details[3]
    p_peaks = []
    p_onsets = []
    p_offsets = []
    max_window = int(fs * 0.3)  # 300 ms maximum window

    # find P wave
    for i_qrs, qrs_onset in enumerate(QRS_onset):
        ibi = interbeat_interval(r_peaks, i_qrs)
        fixed_window = min(ibi, max_window)
        zero_crossings = {}
        # save all the zero crossings in the window
        start_win = qrs_onset-fixed_window
        end_win = qrs_onset
        for i in range(start_win, end_win):
            AMMp, left_peak_pos, right_peak_pos = is_zero_crossing(d3, i)
            if AMMp > 0:
                zero_crossings.update({i: {"AMMp": AMMp, "n_pre": left_peak_pos, "n_post": right_peak_pos}})
        # find P peak
        if len(zero_crossings) > 0: 
            p_peak_minpos = qrs_onset - fixed_window + 0.1*fs
            AMMp = 0
            p_peak = n_pre = n_post = None
            for zc, values in zero_crossings.items():
                if zc > p_peak_minpos and values["AMMp"] > AMMp and values["n_post"] < end_win and values["n_pre"] > start_win:
                    p_peak = zc
                    n_pre = values["n_pre"]
                    n_post = values["n_post"]
                    AMMp = values["AMMp"]
            # if a P peak was found        
            if p_peak is not None:
                # Is it bi-phasic?
                if biphasic_first_condition(d3, n_pre, n_post):
                    # check if it is LEFT bi-phasic
                    # check if there is a zc max 100ms before the peak 
                    left_peak_window = range(int(p_peak - 0.1 * fs), p_peak)
                    for zc, values in zero_crossings.items():
                        if zc in left_peak_window and biphasic_first_condition(d3, values["n_pre"], values["n_post"]):
                            if biphasic_second_condition(AMMp, values["AMMp"]):
                                n_pre = values["n_pre"]
                    ## check if it is RIGHT bi-phasic
                    # check if there is a zc max 100ms after the peak 
                    right_peak_window = range(p_peak, int(p_peak + 0.1 * fs))
                    for zc, values in zero_crossings.items():
                        if zc in right_peak_window and biphasic_first_condition(d3, values["n_pre"], values["n_post"]):
                            if biphasic_second_condition(AMMp, values["AMMp"]):
                                n_post = values["n_post"]
                
                # find P onset
                p_onset_min = qrs_onset-fixed_window
                i = n_pre
                thresh_onset = np.right_shift(abs(d3[n_pre]), 2)        # (6.3) threshold: |d3(n_pre)|/4
                candidate_onset = None
                while i > p_onset_min and candidate_onset is None:
                    if abs(d3[i]) < thresh_onset:
                        candidate_onset = i
                    i -= 1

                # find P offset
                p_offset_max = qrs_onset
                i = n_post
                thresh_offset = np.right_shift(35*abs(d3[n_post]), 6)    # (6.4) threshold: 35*|d3(n_post)|/64
                candidate_offset = None
                while i < p_offset_max and candidate_offset is None:
                    if abs(d3[i]) < thresh_offset:
                        candidate_offset = i
                    i += 1

                if candidate_offset and candidate_onset:
                    p_peaks.append(p_peak)
                    p_onsets.append(candidate_onset)
                    p_offsets.append(candidate_offset)
                else:
                    p_peaks.append(None)
                    p_onsets.append(None)
                    p_offsets.append(None)
            else:
                p_peaks.append(None)
                p_onsets.append(None)
                p_offsets.append(None)
        else:
            p_peaks.append(None)
            p_onsets.append(None)
            p_offsets.append(None)

    return p_peaks, p_onsets, p_offsets