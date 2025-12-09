import numpy as np
import wfdb
import os
import pandas as pd
from scipy.signal import resample
import warnings

# Import your algorithm modules
from integer_wavelet_transform import integer_wavelet_transform
from qrs_detection import qrs_detection
from qrs_delineation import qrs_delineation
from p_delineation import p_delineation
from t_delineation import t_delineation

# ==========================================
# CONFIGURATION
# ==========================================
# Paths to your PhysioNet databases
MITDB_PATH = '/home/utente/bsp_project/BSP-project/data/mit-bih-arrhythmia-database-1.0.0' 
QTDB_PATH = '/home/utente/bsp_project/BSP-project/data/qt-database-1.0.0'
EDB_PATH = '/home/utente/bsp_project/BSP-project/data/european-st-t-database-1.0.0'

# Paper Parameters
TARGET_FS = 250
TOLERANCE_MS = 150  # 150ms tolerance window
LEARNING_PERIOD_SEC = 8  # Ignore first 8 seconds

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def match_annotations(detected, reference, tolerance_samples):
    """
    Matches detected points to reference points within a tolerance.
    Returns: TP, FP, FN, and errors (for delineation)
    """
    # Filter out None values from detected lists
    detected = np.array([d for d in detected if d is not None])
    reference = np.array([r for r in reference if r is not None])
    
    tp_count = 0
    errors = []
    
    # Track which reference points have been matched to avoid double counting
    matched_ref = set()
    
    for d in detected:
        # Find closest reference point
        if len(reference) == 0:
            break
            
        dist = np.abs(reference - d)
        min_idx = np.argmin(dist)
        min_dist = dist[min_idx]
        
        if min_dist <= tolerance_samples:
            if min_idx not in matched_ref:
                tp_count += 1
                matched_ref.add(min_idx)
                errors.append(d - reference[min_idx]) # Error: Detected - Reference
    
    fn_count = len(reference) - tp_count
    fp_count = len(detected) - tp_count
    
    return tp_count, fp_count, fn_count, errors

def load_and_preprocess(record_path, record_name, target_fs):
    """
    Loads record, extracts lead 0, and resamples if necessary.
    """
    try:
        record = wfdb.rdrecord(os.path.join(record_path, record_name), physical=False)
        signal = record.d_signal[:, 0] # Lead 0
        fs = record.fs
        
        # Resample if necessary (e.g. MITDB is 360Hz, paper uses 250Hz)
        if fs != target_fs:
            num_samples = int(len(signal) * target_fs / fs)
            signal = resample(signal, num_samples)
            scaling_factor = target_fs / fs
        else:
            scaling_factor = 1.0
            
        return signal, scaling_factor, record.fs
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return None, None, None

def parse_qtdb_annotations(record_path, record_name, scaling_factor=1.0):
    """
    Parses QTDB .q1c (manually determined boundaries) files.
    """
    try:
        # q1c extension denotes "Manually determined boundaries, annotator 1"
        ann = wfdb.rdann(os.path.join(record_path, record_name), 'q1c')
        
        ref = {
            'P_on': [], 'P_peak': [], 'P_off': [],
            'QRS_on': [], 'QRS_off': [], 
            'T_peak': [], 'T_off': []
        }
        
        # Iterate through symbols to link onsets/offsets with peaks
        samples = ann.sample
        symbols = ann.symbol
        
        for i, sym in enumerate(symbols):
            # Scale sample position if we resampled the signal
            pos = int(samples[i] * scaling_factor)
            
            if sym == 'p':
                ref['P_peak'].append(pos)
                if i > 0 and symbols[i-1] == '(':
                    ref['P_on'].append(int(samples[i-1] * scaling_factor))
                if i < len(symbols)-1 and symbols[i+1] == ')':
                    ref['P_off'].append(int(samples[i+1] * scaling_factor))
                    
            elif sym == 'N': # Normal beat (QRS)
                if i > 0 and symbols[i-1] == '(':
                    ref['QRS_on'].append(int(samples[i-1] * scaling_factor))
                if i < len(symbols)-1 and symbols[i+1] == ')':
                    ref['QRS_off'].append(int(samples[i+1] * scaling_factor))
                    
            elif sym == 't':
                ref['T_peak'].append(pos)
                if i < len(symbols)-1 and symbols[i+1] == ')':
                    ref['T_off'].append(int(samples[i+1] * scaling_factor))
                    
        return ref
    except Exception as e:
        return None

# ==========================================
# MAIN EVALUATION LOGIC
# ==========================================

def run_evaluation():
    tolerance_samples = int(TOLERANCE_MS / 1000 * TARGET_FS)
    learning_samples = LEARNING_PERIOD_SEC * TARGET_FS

    # ---------------------------------------------------------
    # 1. QRS DETECTION VALIDATION (MIT-BIH Arrhythmia Database)
    # ---------------------------------------------------------
    print("="*60)
    print(f"Validating QRS Detection on MITDB (Resampled to {TARGET_FS}Hz)")
    print("="*60)
    
    try:
        mit_records = [f.split('.')[0] for f in os.listdir(MITDB_PATH) if f.endswith('.dat')]
        mit_records.sort()
    except FileNotFoundError:
        print(f"MITDB path not found at {MITDB_PATH}. Skipping.")
        mit_records = []

    mit_results = {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0}
    
    for rec in mit_records:
        if rec == '207': continue # Explicit exclusion as per paper

        # 1. Load Data
        ecg, scaling, original_fs = load_and_preprocess(MITDB_PATH, rec, TARGET_FS)
        if ecg is None: continue

        # 2. Run Algorithm
        details = integer_wavelet_transform(ecg, levels=4)
        r_peaks, _ = qrs_detection(details, TARGET_FS)
        
        # 3. Load Reference Annotations
        ann = wfdb.rdann(os.path.join(MITDB_PATH, rec), 'atr')
        ref_peaks = (ann.sample * scaling).astype(int)
        
        # Filter out annotations/detections during learning period
        ref_peaks = ref_peaks[ref_peaks > learning_samples]
        r_peaks = np.array([r for r in r_peaks if r > learning_samples])

        # 4. Evaluate
        tp, fp, fn, _ = match_annotations(r_peaks, ref_peaks, tolerance_samples)
        
        mit_results['TP'] += tp
        mit_results['FP'] += fp
        mit_results['FN'] += fn
        mit_results['Total_Ann'] += len(ref_peaks)

    # Calculate Global Stats
    if mit_results['TP'] > 0:
        Se = mit_results['TP'] / (mit_results['TP'] + mit_results['FN']) * 100
        Pp = mit_results['TP'] / (mit_results['TP'] + mit_results['FP']) * 100
        print("-" * 40)
        print(f"MITDB Aggregate Results ({mit_results['Total_Ann']} beats):")
        print(f"Sensitivity (Se): {Se:.2f}% (Paper: ~99.78%)")
        print(f"Positive Predictivity (P+): {Pp:.2f}% (Paper: ~99.86%)")
        print("-" * 40)
    else:
        print("No MITDB data processed.")

    # ---------------------------------------------------------
    # 2. QRS DETECTION VALIDATION (European ST-T Database)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"Validating QRS Detection on EDB (Native {TARGET_FS}Hz)")
    print("="*60)

    try:
        edb_records = [f.split('.')[0] for f in os.listdir(EDB_PATH) if f.endswith('.dat')]
        edb_records.sort()
    except FileNotFoundError:
        print(f"EDB path not found at {EDB_PATH}. Skipping.")
        edb_records = []

    edb_results = {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0}

    for rec in edb_records:
        # 1. Load Data
        # EDB is 250Hz, so scaling should be 1.0
        ecg, scaling, original_fs = load_and_preprocess(EDB_PATH, rec, TARGET_FS)
        if ecg is None: continue

        # 2. Run Algorithm
        details = integer_wavelet_transform(ecg, levels=4)
        r_peaks, _ = qrs_detection(details, TARGET_FS)

        # 3. Load Reference Annotations
        try:
            ann = wfdb.rdann(os.path.join(EDB_PATH, rec), 'atr')
            ref_peaks = (ann.sample * scaling).astype(int)
            
            # Filter learning period
            ref_peaks = ref_peaks[ref_peaks > learning_samples]
            r_peaks = np.array([r for r in r_peaks if r > learning_samples])

            # 4. Evaluate
            tp, fp, fn, _ = match_annotations(r_peaks, ref_peaks, tolerance_samples)
            
            edb_results['TP'] += tp
            edb_results['FP'] += fp
            edb_results['FN'] += fn
            edb_results['Total_Ann'] += len(ref_peaks)
        except Exception as e:
            # print(f"Error processing annotations for {rec}: {e}")
            pass

    # Calculate Global Stats
    if edb_results['TP'] > 0:
        Se = edb_results['TP'] / (edb_results['TP'] + edb_results['FN']) * 100
        Pp = edb_results['TP'] / (edb_results['TP'] + edb_results['FP']) * 100
        print("-" * 40)
        print(f"EDB Aggregate Results ({edb_results['Total_Ann']} beats):")
        print(f"Sensitivity (Se): {Se:.2f}% (Paper: ~99.81%)")
        print(f"Positive Predictivity (P+): {Pp:.2f}% (Paper: ~99.56%)")
        print("-" * 40)
    else:
        print("No EDB data processed.")

    # ---------------------------------------------------------
    # 3. DELINEATION VALIDATION (QT Database)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"Validating Delineation on QTDB (Native {TARGET_FS}Hz)")
    print("="*60)

    try:
        qtdb_records = [f.split('.')[0] for f in os.listdir(QTDB_PATH) if f.endswith('.dat') and f.startswith('sel')]
        qtdb_records.sort()
    except FileNotFoundError:
        print(f"QTDB path not found at {QTDB_PATH}. Skipping.")
        qtdb_records = []

    delin_stats = {
        'P_on': [], 'P_peak': [], 'P_off': [],
        'QRS_on': [], 'QRS_off': [],
        'T_peak': [], 'T_off': []
    }

    for rec in qtdb_records:
        ecg, scaling, original_fs = load_and_preprocess(QTDB_PATH, rec, TARGET_FS)
        if ecg is None: continue

        ref = parse_qtdb_annotations(QTDB_PATH, rec, scaling)
        if ref is None: continue 

        # 3. Run Algorithm Chain
        details = integer_wavelet_transform(ecg, levels=4)
        
        # Detection
        r_peaks, qrs_d2_pos = qrs_detection(details, TARGET_FS)
        
        # QRS Delineation
        qrs_on, qrs_off = qrs_delineation(details, TARGET_FS, r_peaks, qrs_d2_pos)
        
        # P Delineation
        p_peaks, p_on, p_off = p_delineation(details, TARGET_FS, qrs_on, r_peaks)
        
        # T Delineation
        t_peaks, t_off, t_ends = t_delineation(details, TARGET_FS, qrs_off, r_peaks)

        # 4. Match and Calculate Errors
        algo_outputs = {
            'P_on': p_on, 'P_peak': p_peaks, 'P_off': p_off,
            'QRS_on': qrs_on, 'QRS_off': qrs_off,
            'T_peak': t_peaks, 'T_off': t_off 
        }

        for key in delin_stats.keys():
            if key in ref and len(ref[key]) > 0:
                det_points = [x for x in algo_outputs[key] if x is not None and x > learning_samples]
                ref_points = [x for x in ref[key] if x > learning_samples]

                if not det_points: continue

                _, _, _, errors_ms = match_annotations(np.array(det_points), np.array(ref_points), tolerance_samples)
                
                # Convert sample errors to milliseconds
                errors_ms = [e / TARGET_FS * 1000 for e in errors_ms]
                
                if len(errors_ms) > 0:
                    delin_stats[key].extend(errors_ms)

    # 5. Print Table
    print("\n{:<12} | {:<15} | {:<15} | {:<15}".format("Fiducial", "Count", "Mean Err (ms)", "Std Dev (ms)"))
    print("-" * 65)
    
    paper_targets = {
        'P_on': '-4.5 ± 13.4', 'P_peak': '-4.7 ± 9.7', 'P_off': '-2.5 ± 13.0',
        'QRS_on': '-5.1 ± 7.2', 'QRS_off': '0.9 ± 8.7',
        'T_peak': '-0.3 ± 12.8', 'T_off': '1.3 ± 18.6'
    }

    for key, errors in delin_stats.items():
        if len(errors) > 0:
            m = np.mean(errors)
            s = np.std(errors)
            print("{:<12} | {:<15} | {:<15.2f} | {:<15.2f} (Target: {})".format(
                key, len(errors), m, s, paper_targets.get(key, "")))
        else:
            print("{:<12} | {:<15} | N/A".format(key, 0))

if __name__ == "__main__":
    run_evaluation()