"""
ECG Utility Functions
Helper functions for loading, processing, and visualizing ECG data
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb
    import plotly.graph_objects as go



def load_ecg(record_name, database_path):
    """
    Load ECG from PhysioNet using WFDB.
    
    Parameters:
    -----------
    record_name : str
        Record identifier (e.g., '100')
    database_path : str
        Path to database directory
    
    Returns:
    --------
    ecg : numpy array
        ECG signal data
    fs : int
        Sampling frequency in Hz
    record : wfdb.Record
        Full record object with metadata
    """
    record = wfdb.rdrecord(f"{database_path}/{record_name}")
    fs = record.fs
    ecg = record.p_signal[:, 0]  # Lead 1 or MLII depending on DB
    return ecg, fs, record


def load_annotations(record_name, database_path, extension='atr'):
    """
    Load annotation file for a record.
    
    Parameters:
    -----------
    record_name : str
        Record identifier
    database_path : str
        Path to database directory
    extension : str
        Annotation file extension (default: 'atr')
    
    Returns:
    --------
    annotation : wfdb.Annotation
        Annotation object with beat locations and types
    """
    annotation = wfdb.rdann(f"{database_path}/{record_name}", extension=extension)
    return annotation


def plot_ecg(ecg, fs, annotations=None, time_range=None, title='ECG Signal'):
    """
    Plot ECG signal with optional annotations overlay.
    
    Parameters:
    -----------
    ecg : numpy array
        ECG signal data
    fs : int
        Sampling frequency in Hz
    annotations : wfdb.Annotation or None
        Optional annotation object with beat locations and types.
        If provided, will overlay beat markers on the plot.
    time_range : tuple or None
        Time range to display as (start, end) in seconds.
        If None, shows the full signal.
    title : str
        Plot title
    """
    # Create time array
    time = np.arange(len(ecg)) / fs
    
    # Plot ECG signal
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time, ecg, 'r-', linewidth=0.5, label='ECG')
    
    # Plot annotations if provided
    if annotations is not None:
        ann_time = annotations.sample / fs
        
        # Filter annotations to time range
        if time_range is not None:
            mask = (ann_time >= time_range[0]) & (ann_time <= time_range[1])
            ann_time_filtered = ann_time[mask]
            ann_symbol_filtered = np.array(annotations.symbol)[mask]
            ann_sample_filtered = annotations.sample[mask]
        else:
            ann_time_filtered = ann_time
            ann_symbol_filtered = annotations.symbol
            ann_sample_filtered = annotations.sample
        
        # Plot annotation markers
        for t, s, sample in zip(ann_time_filtered, ann_symbol_filtered, ann_sample_filtered):
            ax.axvline(t, color='blue', alpha=0.3, linewidth=0.8)
            ax.text(t, ecg[sample], s, fontsize=8, ha='center', va='bottom')
    
    # Add grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='red', alpha=0.3)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.3, color='red', alpha=0.15)
    ax.minorticks_on()
    
    # Labels
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (mV)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set time range
    if time_range is not None:
        ax.set_xlim(time_range[0], time_range[1])
    else:
        ax.set_xlim(0, len(ecg)/fs)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Total duration: {len(ecg)/fs:.1f} seconds")
    print(f"Sampling frequency: {fs} Hz")
    if annotations is not None:
        print(f"Annotations displayed: {len(ann_time_filtered) if time_range else len(annotations.sample)}")


def plot_ecg_interactive(ecg, fs, annotations=None, time_range=None, title='ECG Signal - Interactive'):
    """
    Plot ECG signal with Plotly interactive visualization.
    
    Parameters:
    -----------
    ecg : numpy array
        ECG signal data
    fs : int
        Sampling frequency in Hz
    annotations : wfdb.Annotation or None
        Optional annotation object with beat locations and types.
        If provided, will overlay beat markers on the plot.
    time_range : tuple or None
        Time range to display as (start, end) in seconds.
        If None, shows the full signal.
    title : str
        Plot title
    """    
    # Create time array
    time = np.arange(len(ecg)) / fs
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    
    # Add ECG trace
    fig.add_trace(go.Scatter(
        x=time,
        y=ecg,
        mode='lines',
        name='ECG',
        line=dict(color='red', width=1)
    ))
    
    # Add annotations if provided
    if annotations is not None:
        ann_time = annotations.sample / fs
        
        # Filter annotations to time range
        if time_range is not None:
            mask = (ann_time >= time_range[0]) & (ann_time <= time_range[1])
            ann_time_filtered = ann_time[mask]
            ann_symbol_filtered = np.array(annotations.symbol)[mask]
            ann_sample_filtered = annotations.sample[mask]
        else:
            ann_time_filtered = ann_time
            ann_symbol_filtered = annotations.symbol
            ann_sample_filtered = annotations.sample
        
        # Add vertical lines for each annotation
        for t, s, sample in zip(ann_time_filtered, ann_symbol_filtered, ann_sample_filtered):
            fig.add_vline(
                x=t,
                line=dict(color='blue', width=1, dash='dash'),
                opacity=0.5,
                annotation_text=s,
                annotation_position='top'
            )
    
    fig.update_layout(
        title=f'{title} (Drag to zoom, double-click to reset)',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude (mV)',
        height=500,
        hovermode='x unified',
        xaxis=dict(
            gridcolor='rgba(255,0,0,0.2)',
            rangeslider=dict(visible=True)  # Bottom slider for navigation
        ),
        yaxis=dict(gridcolor='rgba(255,0,0,0.2)')
    )
    
    # Set time range if specified
    if time_range is not None:
        fig.update_xaxes(range=[time_range[0], time_range[1]])
    
    fig.show()
    
    print(f"Total duration: {len(ecg)/fs:.1f} seconds")
    print(f"Sampling frequency: {fs} Hz")
    if annotations is not None:
        print(f"Annotations displayed: {len(ann_time_filtered) if time_range else len(annotations.sample)}")
