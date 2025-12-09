import numpy as np

def remove_group_delay(details):
    '''
    Remove group delay from detail coefficients. 
    Using equations (4.1), (4.2), (4.3) from the paper.
    '''
    for k in details.keys():
        if k == 1:
            # For k=1: tau_1_G = -2^(1-2) = -2^(-1) = -0.5 ≈ 0 (rounds to 0)
            tau = -2**(1 - 2)  # This equals -0.5
        else:
            # k >= 2: sum of H filter delays + G filter delay
            tau = 0
            # Sum H filter delays from l=1 to k-1
            for l in range(1, k):
                tau_H = -3 * 2**(l - 2)
                tau += tau_H
            # Add G filter delay at scale k
            tau_G = -2**(k - 2)
            tau += tau_G
        
        # Negative tau means shift right (delay), so we shift left to compensate
        shift = int(np.round(tau))
        if shift != 0:
            details[k] = np.roll(details[k], shift)
    
    return details

def integer_wavelet_transform(ecg_signal, levels=4):
    '''
    Perform Integer Wavelet Transform using the "algorithme à trous" (undecimated DWT).
    Implements the filter bank from Di Marco & Chiari (2011), Figure 2, Table 1.
    
    Parameters:
    -----------
    ecg_signal : array-like
        Raw ECG signal (assumed to be in integer format, e.g., 12-bit ADC values)
    levels : int
        Number of decomposition levels (default 4 for scales 2^1 to 2^4)
    
    Returns:
    --------
    details : dict
        Dictionary of detail coefficients at each scale k (key = k, value = d^k_n)
    '''
    # Convert to 32-bit integers to prevent overflow
    signal = np.array(ecg_signal, dtype=np.int32)
    N = len(signal)
    
    # Filter coefficients from Table 1
    # store numerators and divide by 128 after convolution
    h_num = np.array([1, 7, 21, 35, 35, 21, 7, 1], dtype=np.int32)
    g_num = np.array([-2, 2], dtype=np.int32)
    
    details = {}
    approximation = signal.copy()
    
    for k in range(1, levels + 1):
        # Dilation factor: insert (2^(k-1) - 1) zeros between coefficients
        dilation = 2**(k - 1)
        
        # === Create dilated filters (algorithme à trous) ===
        # H filter (low-pass)
        h_k_len = len(h_num) + (len(h_num) - 1) * (dilation - 1)
        h_k = np.zeros(h_k_len, dtype=np.int32)
        h_k[::dilation] = h_num
        
        # G filter (high-pass)
        g_k_len = len(g_num) + (len(g_num) - 1) * (dilation - 1)
        g_k = np.zeros(g_k_len, dtype=np.int32)
        g_k[::dilation] = g_num
        
        # === Apply filters ===
        # Convolve with H to get next approximation
        a_conv = np.convolve(approximation, h_k, mode='full')
        # For h starting at n=-2, offset = 2 * dilation
        offset_h = 2 * dilation
        start_h = offset_h
        a_next_num = a_conv[start_h:start_h + N]
        
        # Divide by 128 for H filter (from Table 1 denominators)
        a_next = np.right_shift(a_next_num, 7)  # 128 = 2^7
        
        # Convolve with G to get detail coefficients
        d_conv = np.convolve(approximation, g_k, mode='full')
        # No offset needed for g since it starts at 0
        start_g = 0
        d_next = d_conv[start_g:start_g + N]
        
        # Handle edge artifacts by setting boundary values to mean
        # The paper doesn't specify but is reasonable
        boundary = max(h_k_len, g_k_len) // 2
        if boundary > 0:
            d_mean = np.mean(d_next[boundary:-boundary]) if N > 2*boundary else 0
            d_next[:boundary] = d_mean
            d_next[-boundary:] = d_mean
        
        # Store detail coefficients
        details[k] = d_next.astype(np.int32)
        
        # Update approximation for next iteration
        approximation = a_next.astype(np.int32)
    
    # Apply group delay correction to align zero-crossings across scales
    details = remove_group_delay(details)
    
    return details