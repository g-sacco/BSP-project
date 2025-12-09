from scipy.signal import resample

def preprocessing(ecg, fs, target_fs=250):
    # downsample ecg signal
    ecg = resample(ecg, int(len(ecg) * target_fs / fs))
    return ecg, target_fs