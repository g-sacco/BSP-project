from scipy.signal import resample
import wfdb
import os

def load_and_preprocess(record_path, record_name, target_fs):
    """
    Loads record, extracts lead 0 and 1 (integer values), and resamples if necessary.
    """
    try:
        record = wfdb.rdrecord(os.path.join(record_path, record_name), physical=False)
        signal_0 = record.d_signal[:, 0] # Lead 0
        signal_1 = record.d_signal[:, 1] # Lead 1
        fs = record.fs
        
        # Resample if necessary (e.g. MITDB is 360Hz, paper uses 250Hz)
        if fs != target_fs:
            num_samples = int(len(signal_0) * target_fs / fs)
            signal_0 = resample(signal_0, num_samples)
            signal_1 = resample(signal_1, num_samples)
            scaling_factor = target_fs / fs
        else:
            scaling_factor = 1.0
            
        return signal_0, signal_1, scaling_factor, record.fs
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return None, None, None, None
    
def load_and_preprocess_physical(record_path, record_name, target_fs):
    """
    Loads record, extracts lead 0 and 1 (physical values), and resamples if necessary.
    """
    try:
        record = wfdb.rdrecord(os.path.join(record_path, record_name), physical=True)
        signal_0 = record.p_signal[:, 0] # Lead 0
        signal_1 = record.p_signal[:, 1] # Lead 1
        fs = record.fs
        
        # Resample if necessary (e.g. MITDB is 360Hz, paper uses 250Hz)
        if fs != target_fs:
            num_samples = int(len(signal_0) * target_fs / fs)
            signal_0 = resample(signal_0, num_samples)
            signal_1 = resample(signal_1, num_samples)
            scaling_factor = target_fs / fs
        else:
            scaling_factor = 1.0
            
        return signal_0, signal_1, scaling_factor, record.fs
    except Exception as e:
        print(f"Error loading {record_name}: {e}")
        return None, None, None, None