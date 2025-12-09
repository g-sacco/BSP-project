import numpy as np
from qrs_detection import is_zero_crossing

def t_delineation(details, fs, QRS_offsets, r_peaks):
    d3 = details[3]
    DW80 = int(0.08 * fs)
    t_peaks = []
    t_offsets = []
    t_ends = []
    # i = QRS counter
    # j = sample in the current window
    for i in range(1, len(QRS_offsets)):
        t_peak = t_off = t_end = None
        RR = r_peaks[i] - r_peaks[i-1]
        T_win = (QRS_offsets[i-1] + DW80, QRS_offsets[i-1] + np.right_shift(19*RR, 5))
        Tpeak_win = (T_win[0], QRS_offsets[i-1] + np.right_shift(RR, 1))
        # save all the zero crossings in the window
        zcs = {}
        max_zc_AMMp = - np.inf
        prev_peak = None
        next_peak = None
        
        for j in range(Tpeak_win[0], Tpeak_win[1]):
            AMMp, left_peak_pos, right_peak_pos = is_zero_crossing(d3, j)
            if AMMp > 0:
                zcs.update({j: {"AMMp": AMMp, "n_pre": left_peak_pos, "n_post": right_peak_pos}})
                if max_zc_AMMp < AMMp:
                    prev_peak = t_peak          # save the previous peak before updating
                    t_peak = j
                    max_zc_AMMp = AMMp
                    next_peak = None            # reset next_peak when we find a new max
                elif t_peak is not None and next_peak is None:
                    next_peak = j               # first peak after the maximum
    
        if t_peak:
            # if the t wave is (left) bi-phasic
            bi_thresh_left = np.right_shift(51*zcs[t_peak]["AMMp"], 6)
            if prev_peak is not None and zcs[prev_peak]["AMMp"] > bi_thresh_left:
                t_off = t_peak
                t_peak = prev_peak     # the first peak is always the main one
                # calculate T end
                n_post = zcs[t_off]["n_post"]
                thresh_end = np.right_shift(19*abs(d3[n_post]), 6)
                j = n_post
                while j < T_win[1] and t_end is None:
                    if abs(d3[j]) < thresh_end:
                        t_end = j
                    j += 1
            # if the t wave is (right) bi-phasic or mono-phasic
            else:
                # find t offset
                n_post = zcs[t_peak]["n_post"]
                thresh_off = np.right_shift(19*abs(d3[n_post]), 6)
                j = n_post
                while j < T_win[1] and t_off is None:
                    if abs(d3[j]) < thresh_off:
                        t_off = j
                    j += 1
                # if there is a right peak
                if next_peak is not None:
                    # if that peak is big enough then it is bi-phasic
                    bi_thresh_right = np.right_shift(zcs[t_peak]["AMMp"], 1)
                    if zcs[next_peak]["AMMp"] >= bi_thresh_right:
                        # find t end after the second peak
                        n_post = zcs[next_peak]["n_post"]
                        thresh_end = np.right_shift(19*abs(d3[n_post]), 6)
                        j = n_post
                        while j < T_win[1] and t_end is None:
                            if abs(d3[j]) < thresh_end:
                                t_end = j
                            j += 1
        t_peaks.append(t_peak)
        t_offsets.append(t_off)
        t_ends.append(t_end)
        
    #print(f"Number of T-waves delineation succeeded: {sum(1 for t in t_peaks if t is not None)}")

    return t_peaks, t_offsets, t_ends