# Export all databases
# python src/results.py --export

# Export specific databases
# python src/results.py MITDB QTDB --export

# Run without export (original behavior)
# python src/results.py QTDB


import numpy as np
import wfdb
import os
import pandas as pd

# Import your algorithm modules
from integer_wavelet_transform import integer_wavelet_transform
from qrs_detection import qrs_detection
from qrs_delineation import qrs_delineation
from p_delineation import p_delineation
from t_delineation import t_delineation
from load_and_preprocess import load_and_preprocess

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

def match_annotations(detected, reference, tolerance_samples, count_all_fps=True):
    """
    Matches detected points to reference points within a tolerance.
    Returns: TP, FP, FN, and errors (for delineation)
    
    Args:
        detected: Array of detected points
        reference: Array of reference annotations
        tolerance_samples: Tolerance window in samples
        count_all_fps: If True, count all unmatched detections as FP.
                       If False, only count FP if detection is within tolerance of any reference.
    """
    # Filter out None values from detected lists
    detected = np.array([d for d in detected if d is not None])
    reference = np.array([r for r in reference if r is not None])
    
    tp_count = 0
    errors = []
    
    # Track which reference points have been matched to avoid double counting
    matched_ref = set()
    matched_det = set()
    
    for i, d in enumerate(detected):
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
                matched_det.add(i)
                errors.append(d - reference[min_idx]) # Error: Detected - Reference
    
    fn_count = len(reference) - tp_count
    
    if count_all_fps:
        # Count all unmatched detections as FP
        fp_count = len(detected) - tp_count
    else:
        # Only count FP if detection is within tolerance of any reference (but already matched)
        # This is for sparse annotations like QTDB
        fp_count = 0
        for i, d in enumerate(detected):
            if i not in matched_det:
                # Check if this detection is within tolerance of ANY reference
                if len(reference) > 0:
                    min_dist_to_any_ref = np.min(np.abs(reference - d))
                    if min_dist_to_any_ref <= tolerance_samples:
                        fp_count += 1
    
    return tp_count, fp_count, fn_count, errors

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

def run_evaluation(databases=['MITDB', 'EDB', 'QTDB'], export_to_files=False):
    tolerance_samples = int(TOLERANCE_MS / 1000 * TARGET_FS)
    learning_samples = LEARNING_PERIOD_SEC * TARGET_FS
    
    # Create results directory if it doesn't exist
    results_dir = '/home/utente/bsp_project/BSP-project/results'
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. QRS DETECTION VALIDATION (MIT-BIH Arrhythmia Database)
    # ---------------------------------------------------------
    if 'MITDB' in databases:
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
        mit_per_record = []  # For exporting per-record results
        
        for rec in mit_records:
            if rec == '207': continue # Explicit exclusion as per paper

            # 1. Load Data
            ecg, _, scaling, original_fs = load_and_preprocess(MITDB_PATH, rec, TARGET_FS)
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
            
            # Store per-record results
            se_rec = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
            pp_rec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
            mit_per_record.append({
                'Record': rec,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'Total_Annotations': len(ref_peaks),
                'Sensitivity_%': se_rec,
                'Positive_Predictivity_%': pp_rec
            })

        # Calculate Global Stats
        if mit_results['TP'] > 0:
            Se = mit_results['TP'] / (mit_results['TP'] + mit_results['FN']) * 100
            Pp = mit_results['TP'] / (mit_results['TP'] + mit_results['FP']) * 100
            print("-" * 40)
            print(f"MITDB Aggregate Results ({mit_results['Total_Ann']} beats):")
            print(f"Sensitivity (Se): {Se:.2f}% (Paper: ~99.78%)")
            print(f"Positive Predictivity (P+): {Pp:.2f}% (Paper: ~99.86%)")
            print("-" * 40)
            
            # Export to CSV if requested
            if export_to_files:
                df_mit = pd.DataFrame(mit_per_record)
                # Add aggregate row
                df_mit = pd.concat([df_mit, pd.DataFrame([{
                    'Record': 'AGGREGATE',
                    'TP': mit_results['TP'],
                    'FP': mit_results['FP'],
                    'FN': mit_results['FN'],
                    'Total_Annotations': mit_results['Total_Ann'],
                    'Sensitivity_%': Se,
                    'Positive_Predictivity_%': Pp
                }])], ignore_index=True)
                csv_path = os.path.join(results_dir, 'MITDB_results.csv')
                df_mit.to_csv(csv_path, index=False)
                print(f"Results exported to {csv_path}")
        else:
            print("No MITDB data processed.")    
            
    # ---------------------------------------------------------
    # 2. QRS DETECTION VALIDATION (European ST-T Database)
    # ---------------------------------------------------------
    if 'EDB' in databases:
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
        edb_per_record = []  # For exporting per-record results

        for rec in edb_records:
            # 1. Load Data
            # EDB is 250Hz, so scaling should be 1.0
            ecg, _, scaling, original_fs = load_and_preprocess(EDB_PATH, rec, TARGET_FS)
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
                
                # Store per-record results
                se_rec = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
                pp_rec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
                edb_per_record.append({
                    'Record': rec,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'Total_Annotations': len(ref_peaks),
                    'Sensitivity_%': se_rec,
                    'Positive_Predictivity_%': pp_rec
                })
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
            
            # Export to CSV if requested
            if export_to_files:
                df_edb = pd.DataFrame(edb_per_record)
                # Add aggregate row
                df_edb = pd.concat([df_edb, pd.DataFrame([{
                    'Record': 'AGGREGATE',
                    'TP': edb_results['TP'],
                    'FP': edb_results['FP'],
                    'FN': edb_results['FN'],
                    'Total_Annotations': edb_results['Total_Ann'],
                    'Sensitivity_%': Se,
                    'Positive_Predictivity_%': Pp
                }])], ignore_index=True)
                csv_path = os.path.join(results_dir, 'EDB_results.csv')
                df_edb.to_csv(csv_path, index=False)
                print(f"Results exported to {csv_path}")
        else:
            print("No EDB data processed.")    
            
    # ---------------------------------------------------------
    # 3. DELINEATION VALIDATION (QT Database)
    # ---------------------------------------------------------
    if 'QTDB' in databases:
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
        
        qtdb_detection_stats = {
            'P_on': {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0},
            'P_peak': {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0},
            'P_off': {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0},
            'QRS_on': {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0},
            'QRS_off': {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0},
            'T_peak': {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0},
            'T_off': {'TP': 0, 'FP': 0, 'FN': 0, 'Total_Ann': 0}
        }
        
        qtdb_per_record = []  # For exporting per-record results

        for rec in qtdb_records:
            #print(f"\nProcessing QTDB record: {rec}")
            ecg_0, ecg_1, scaling, original_fs = load_and_preprocess(QTDB_PATH, rec, TARGET_FS)
            if ecg_0 is None: continue

            ref = parse_qtdb_annotations(QTDB_PATH, rec, scaling)
            if ref is None: continue 

            # Process both leads and track their errors
            lead_errors = [[], []]  # [ecg_0_errors, ecg_1_errors]
            lead_outputs = []
            
            for lead_idx, ecg in enumerate([ecg_0, ecg_1]):
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
                lead_outputs.append(algo_outputs)
                
                # Calculate total error for this lead
                total_error = 0
                for key in delin_stats.keys():
                    if key in ref and len(ref[key]) > 0:
                        det_points = [x for x in algo_outputs[key] if x is not None and x > learning_samples]
                        ref_points = [x for x in ref[key] if x > learning_samples]

                        if det_points:
                            _, _, _, errors_samples = match_annotations(np.array(det_points), np.array(ref_points), tolerance_samples)
                            errors_ms = [e / TARGET_FS * 1000 for e in errors_samples]
                            if errors_ms:
                                # Sum absolute errors for comparison
                                total_error += sum(abs(e) for e in errors_ms)
                                lead_errors[lead_idx].extend(errors_ms)
            
            # Choose the lead with the least total absolute error
            if lead_errors[0] or lead_errors[1]:
                total_err_0 = sum(abs(e) for e in lead_errors[0]) if lead_errors[0] else float('inf')
                total_err_1 = sum(abs(e) for e in lead_errors[1]) if lead_errors[1] else float('inf')
                
                best_lead = 0 if total_err_0 <= total_err_1 else 1
                best_outputs = lead_outputs[best_lead]
                
                # Add errors from best lead to statistics
                for key in delin_stats.keys():
                    if key in ref and len(ref[key]) > 0:
                        det_points = [x for x in best_outputs[key] if x is not None and x > learning_samples]
                        ref_points = [x for x in ref[key] if x > learning_samples]

                        if det_points:
                            # For QTDB, only count FPs near annotated beats (sparse annotations)
                            tp, fp, fn, errors_samples = match_annotations(
                                np.array(det_points), np.array(ref_points), 
                                tolerance_samples, count_all_fps=False
                            )
                            errors_ms = [e / TARGET_FS * 1000 for e in errors_samples]
                            if errors_ms:
                                delin_stats[key].extend(errors_ms)
                            
                            # Accumulate detection statistics per annotation type
                            qtdb_detection_stats[key]['TP'] += tp
                            qtdb_detection_stats[key]['FP'] += fp
                            qtdb_detection_stats[key]['FN'] += fn
                            qtdb_detection_stats[key]['Total_Ann'] += len(ref_points)
                
                # Store per-record results
                record_data = {'Record': rec}
                for key in delin_stats.keys():
                    if key in ref and len(ref[key]) > 0:
                        det_points = [x for x in best_outputs[key] if x is not None and x > learning_samples]
                        ref_points = [x for x in ref[key] if x > learning_samples]
                        if det_points:
                            tp, fp, fn, errors_samples = match_annotations(
                                np.array(det_points), np.array(ref_points), 
                                tolerance_samples, count_all_fps=False
                            )
                            errors_ms = [e / TARGET_FS * 1000 for e in errors_samples]
                            mean_err = np.mean(errors_ms) if errors_ms else None
                            std_err = np.std(errors_ms) if errors_ms else None
                            
                            record_data[f'{key}_Count'] = len(errors_ms)
                            record_data[f'{key}_Mean_ms'] = mean_err
                            record_data[f'{key}_Std_ms'] = std_err
                            record_data[f'{key}_TP'] = tp
                            record_data[f'{key}_FP'] = fp
                            record_data[f'{key}_FN'] = fn
                        else:
                            record_data[f'{key}_Count'] = 0
                            record_data[f'{key}_Mean_ms'] = None
                            record_data[f'{key}_Std_ms'] = None
                            record_data[f'{key}_TP'] = 0
                            record_data[f'{key}_FP'] = 0
                            record_data[f'{key}_FN'] = 0
                    else:
                        record_data[f'{key}_Count'] = 0
                        record_data[f'{key}_Mean_ms'] = None
                        record_data[f'{key}_Std_ms'] = None
                        record_data[f'{key}_TP'] = 0
                        record_data[f'{key}_FP'] = 0
                        record_data[f'{key}_FN'] = 0
                qtdb_per_record.append(record_data)

        # 5. Print Detection Statistics per Annotation Type
        print("\n" + "="*60)
        print("QTDB Detection Statistics per Annotation Type")
        print("="*60)
        print("{:<12} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10}".format(
            "Fiducial", "TP", "FP", "FN", "Total", "Se (%)", "P+ (%)"))
        print("-" * 80)
        
        for key in delin_stats.keys():
            stats = qtdb_detection_stats[key]
            if stats['TP'] > 0:
                Se = stats['TP'] / (stats['TP'] + stats['FN']) * 100
                Pp = stats['TP'] / (stats['TP'] + stats['FP']) * 100
                print("{:<12} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10.2f} | {:<10.2f}".format(
                    key, stats['TP'], stats['FP'], stats['FN'], stats['Total_Ann'], Se, Pp))
            else:
                print("{:<12} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10}".format(
                    key, stats['TP'], stats['FP'], stats['FN'], stats['Total_Ann'], "N/A", "N/A"))
        
        # 6. Print Delineation Error Table
        print("\n" + "="*60)
        print("QTDB Delineation Error Statistics")
        print("="*60)
        print("{:<12} | {:<15} | {:<15} | {:<15}".format("Fiducial", "Count", "Mean Err (ms)", "Std Dev (ms)"))
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
        # Export to CSV if requested
        if export_to_files and qtdb_per_record:
            # Export per-record results
            df_qtdb = pd.DataFrame(qtdb_per_record)
            csv_path = os.path.join(results_dir, 'QTDB_per_record_results.csv')
            df_qtdb.to_csv(csv_path, index=False)
            print(f"\nPer-record results exported to {csv_path}")
            
            # Export detection statistics
            detection_data = []
            for key in delin_stats.keys():
                stats = qtdb_detection_stats[key]
                if stats['TP'] > 0:
                    Se = stats['TP'] / (stats['TP'] + stats['FN']) * 100
                    Pp = stats['TP'] / (stats['TP'] + stats['FP']) * 100
                else:
                    Se = None
                    Pp = None
                detection_data.append({
                    'Fiducial': key,
                    'TP': stats['TP'],
                    'FP': stats['FP'],
                    'FN': stats['FN'],
                    'Total_Annotations': stats['Total_Ann'],
                    'Sensitivity_%': Se,
                    'Positive_Predictivity_%': Pp
                })
            df_detection = pd.DataFrame(detection_data)
            csv_path = os.path.join(results_dir, 'QTDB_detection_statistics.csv')
            df_detection.to_csv(csv_path, index=False)
            print(f"Detection statistics exported to {csv_path}")
            
            # Export delineation error statistics
            error_data = []
            for key, errors in delin_stats.items():
                if len(errors) > 0:
                    m = np.mean(errors)
                    s = np.std(errors)
                else:
                    m = None
                    s = None
                error_data.append({
                    'Fiducial': key,
                    'Count': len(errors),
                    'Mean_Error_ms': m,
                    'Std_Dev_ms': s,
                    'Paper_Target': paper_targets.get(key, '')
                })
            df_errors = pd.DataFrame(error_data)
            csv_path = os.path.join(results_dir, 'QTDB_delineation_errors.csv')
            df_errors.to_csv(csv_path, index=False)
            print(f"Delineation error statistics exported to {csv_path}")
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for datbase selection
    # Usage: python results.py [MITDB] [EDB] [QTDB] [--export | -e]
    # Example: python results.py MITDB QTDB --export (runs only MITDB and QTDB and exports to CSV)
    # If no arguments provided, runs all databases without exporting
    
    # Check if --export flag is present
    export_flag = '--export' in sys.argv or '-e' in sys.argv
    
    # Remove flags from argv before processing database names
    db_args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    
    if len(db_args) > 0:
        # Run only specified databases
        valid_dbs = ['MITDB', 'EDB', 'QTDB']
        selected_dbs = [db.upper() for db in db_args if db.upper() in valid_dbs]
        
        if not selected_dbs:
            print("No valid databases specified. Valid options: MITDB, EDB, QTDB")
            print("Use --export or -e flag to export results to CSV files")
            print("Running all databases by default...")
            run_evaluation(export_to_files=export_flag)
        else:
            print(f"Running evaluation on: {', '.join(selected_dbs)}")
            if export_flag:
                print("Results will be exported to CSV files")
            run_evaluation(databases=selected_dbs, export_to_files=export_flag)
    else:
        # Run all databases by default
        if export_flag:
            print("Results will be exported to CSV files")
        run_evaluation(export_to_files=export_flag)

