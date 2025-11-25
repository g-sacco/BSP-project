"""
ECG Delineation Algorithm (Di Marco & Chiari 2011)
Python Skeleton

Structure:
1) Wavelet decomposition (SWT)
2) QRS detection
3) QRS onset/offset
4) P-wave detection
5) T-wave detection
6) Performance evaluation
"""

import numpy as np
import pywt
from scipy.signal import find_peaks
import wfdb

# ---------------------------------------------------------------------
# 0. LOADING DATA (use WFDB library)
# ---------------------------------------------------------------------
def load_ecg(record_name, database_path):
    """
    Load ECG from PhysioNet using WFDB.
    Returns:
        ecg: numpy array
        fs: sampling frequency
    """
    record = wfdb.rdrecord(f"{database_path}/{record_name}")
    fs = record.fs
    ecg = record.p_signal[:, 0]  # Lead 1 or MLII depending on DB
    return ecg, fs


# ---------------------------------------------------------------------
# 1. STATIONARY WAVELET TRANSFORM (algorithme à trous)
# ---------------------------------------------------------------------
def wavelet_decomposition(ecg, wavelet="sym4", level=3):
    """
    Perform SWT (Stationary Wavelet Transform).
    According to the paper:
        scale 2^2  -> zero crossings for QRS
        scale 2^3  -> modulus maxima for amplitude

    Returns:
        coeffs: list of (cA_i, cD_i) pairs
    """
    coeffs = pywt.swt(ecg, wavelet, level=level, trim_approx=False)
    return coeffs


# ---------------------------------------------------------------------
# 2. ZERO-CROSSINGS AND MODULUS MAXIMA (QRS candidates)
# ---------------------------------------------------------------------
def detect_zero_crossings(coeff_scale_2):
    """
    Detect zero crossings on SWT detail coefficients at scale 2^2.
    Returns:
        zc_locs: indices of zero crossings
    """
    sign_changes = np.diff(np.sign(coeff_scale_2))
    zc_locs = np.where(sign_changes != 0)[0]
    return zc_locs


def compute_modulus_maxima(coeff_scale_3):
    """
    Compute modulus maxima pairs (AMMp).
    According to the paper, these help measure QRS amplitude.
    """
    # absolute peaks for modulus maxima
    peaks, _ = find_peaks(np.abs(coeff_scale_3))
    return peaks


# ---------------------------------------------------------------------
# 3. QRS DETECTION LOGIC (Di Marco & Chiari Section 5)
# ---------------------------------------------------------------------
def detect_qrs(ecg, coeffs, fs):
    """
    Main QRS detection algorithm.
    Steps:
    1. zero crossings at scale 2^2
    2. modulus maxima at scale 2^3
    3. sliding window 250 ms
    4. adaptive thresholds ε2_QRS, ε3_QRS
    5. decision rule → R-peaks

    Returns:
        r_peaks: list of R peak sample indices
    """

    # Extract scales following paper logic
    _, cD2 = coeffs[1]    # scale 2^2
    _, cD3 = coeffs[2]    # scale 2^3

    zc = detect_zero_crossings(cD2)
    maxima = compute_modulus_maxima(cD3)

    # Placeholder threshold values (TO BE IMPLEMENTED)
    eps2 = np.mean(np.abs(cD2)) * 0.5
    eps3 = np.mean(np.abs(cD3)) * 0.5

    r_peaks = []

    # Sliding window: 250 ms
    window = int(0.25 * fs)

    for z in zc:
        # find modulus maxima near zero crossing
        nearby = maxima[(maxima > z - window) & (maxima < z + window)]
        if len(nearby) == 0:
            continue

        # amplitude measure (placeholder)
        amp = np.max(np.abs(cD3[nearby]))

        # decision rule from the paper (TO BE IMPLEMENTED)
        if np.abs(cD2[z]) > eps2 and amp > eps3:
            r_peaks.append(z)

    return np.array(r_peaks)


# ---------------------------------------------------------------------
# 4. QRS ONSET & OFFSET
# ---------------------------------------------------------------------
def qrs_onset_offset(ecg, r_peaks, coeffs, fs):
    """
    Implement equations 5.6–5.8 from the paper.
    Returns:
        qrs_onsets, qrs_offsets
    """
    qrs_onsets = []
    qrs_offsets = []

    for r in r_peaks:
        # placeholder logic → needs exact rules from paper
        onset = r - int(0.04 * fs)
        offset = r + int(0.04 * fs)

        onset = max(onset, 0)
        offset = min(offset, len(ecg)-1)

        qrs_onsets.append(onset)
        qrs_offsets.append(offset)

    return np.array(qrs_onsets), np.array(qrs_offsets)


# ---------------------------------------------------------------------
# 5. P-WAVE DETECTION
# ---------------------------------------------------------------------
def detect_p_wave(ecg, r_peaks, coeffs, fs):
    """
    Implement Section 6 of the paper.
    Use:
        - search window before QRS
        - maxima/minima rules
        - amplitude thresholds

    Returns:
        p_peaks, p_onsets, p_offsets
    """
    p_peaks = []
    p_onsets = []
    p_offsets = []

    for r in r_peaks:
        # typical window: 100–200 ms before QRS
        start = max(0, r - int(0.2 * fs))
        end = max(0, r - int(0.05 * fs))

        segment = ecg[start:end]
        if len(segment) == 0:
            continue

        # placeholder: simple peak detection
        peak = np.argmax(segment) + start
        p_peaks.append(peak)

        # placeholder onset/offset
        p_onsets.append(peak - int(0.02 * fs))
        p_offsets.append(peak + int(0.02 * fs))

    return np.array(p_peaks), np.array(p_onsets), np.array(p_offsets)


# ---------------------------------------------------------------------
# 6. T-WAVE DETECTION
# ---------------------------------------------------------------------
def detect_t_wave(ecg, qrs_offsets, coeffs, fs):
    """
    Implement Section 7 of the paper.
    Use:
        - search window after QRS offset
        - peak polarity rules
        - offset rules from SWT

    Returns:
        t_peaks, t_offsets
    """
    t_peaks = []
    t_offsets = []

    for qoff in qrs_offsets:
        # search window: 80–400 ms after QRS offset
        start = min(len(ecg)-1, qoff + int(0.08 * fs))
        end = min(len(ecg)-1, qoff + int(0.4 * fs))

        segment = ecg[start:end]
        if len(segment) == 0:
            continue

        # placeholder: simple maximum
        peak = np.argmax(segment) + start
        t_peaks.append(peak)

        # placeholder offset
        t_offsets.append(peak + int(0.04 * fs))

    return np.array(t_peaks), np.array(t_offsets)


# ---------------------------------------------------------------------
# 7. PERFORMANCE EVALUATION
# ---------------------------------------------------------------------
def evaluate_qrs(r_peaks, reference_ann, fs):
    """
    Compute:
        - Sensitivity
        - Positive Predictive Value
    using a matching window ~150 ms.

    Returns:
        se, ppv
    """
    tol = int(0.15 * fs)

    tp = 0
    fp = 0
    fn = 0

    for ann in reference_ann:
        if np.any(np.abs(r_peaks - ann) < tol):
            tp += 1
        else:
            fn += 1

    for r in r_peaks:
        if not np.any(np.abs(reference_ann - r) < tol):
            fp += 1

    se = tp / (tp + fn)
    ppv = tp / (tp + fp)

    return se, ppv


# ---------------------------------------------------------------------
# 8. PIPELINE
# ---------------------------------------------------------------------
def run_pipeline(record_name, db):
    ecg, fs = load_ecg(record_name, db)
    coeffs = wavelet_decomposition(ecg)

    r_peaks = detect_qrs(ecg, coeffs, fs)
    qrs_on, qrs_off = qrs_onset_offset(ecg, r_peaks, coeffs, fs)
    p_peak, p_on, p_off = detect_p_wave(ecg, r_peaks, coeffs, fs)
    t_peak, t_off = detect_t_wave(ecg, qrs_off, coeffs, fs)

    return {
        "R": r_peaks,
        "QRS_on": qrs_on,
        "QRS_off": qrs_off,
        "P": p_peak,
        "P_on": p_on,
        "P_off": p_off,
        "T": t_peak,
        "T_off": t_off
    }


# ---------------------------------------------------------------------
# RUN EXAMPLE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    results = run_pipeline("100", "data/mit-bih-arrhythmia-database-1.0.0")
    print(results)
