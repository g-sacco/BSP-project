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

def load_ecg_int(record_name, database_path):
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
        ECG signal data as integers
    fs : int
        Sampling frequency in Hz
    record : wfdb.Record
        Full record object with metadata
    """
    record = wfdb.rdrecord(f"{database_path}/{record_name}", physical=False)
    fs = record.fs
    ecg = record.d_signal[:, 0]  # Lead 1 or MLII depending on DB (digital/integer values)
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


def plot_ecg(ecg, fs, annotations=None, time_range=None, title='ECG Signal', 
             p_events=None, qrs_events=None, t_events=None, figsize=(10, 5), 
             qrs_d2_peaks_pos=None):
    """
    Plot ECG signal with optional annotations and P-QRS-T events overlay.
    
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
    p_events : dict or None
        P wave events with keys: 'onset', 'peak', 'offset' (sample indices)
    qrs_events : dict or None
        QRS events with keys: 'onset', 'peak', 'offset' (sample indices)
    t_events : dict or None
        T wave events with keys: 'onset', 'peak', 'offset' (sample indices)
    figsize : tuple
        Figure size as (width, height) in inches. Default is (10, 5).
    qrs_d2_peaks_pos : list of tuples or None
        List of (R_pre, R_post) tuples containing sample indices for QRS peak positions
        detected from wavelet detail level 2. Will be plotted as vertical lines.
    """
    # Create time array
    time = np.arange(len(ecg)) / fs
    
    # Plot ECG signal
    fig, ax = plt.subplots(figsize=figsize)
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
    
    # Plot P-QRS-T events
    event_configs = [
        (p_events, 'P', 'green', {'onset': '--', 'peak': '-', 'offset': '--'}),
        (qrs_events, 'QRS', 'black', {'onset': '--', 'peak': '-', 'offset': '--'}),
        (t_events, 'T', 'blue', {'peak': '-', 'offset': '--', 'end': ':'})
    ]
    
    for events, label, color, linestyles in event_configs:
        if events is not None:
            for event_type in linestyles.keys():
                if event_type in events and events[event_type] is not None:
                    # Filter out None values
                    event_list = [e for e in events[event_type] if e is not None]
                    if len(event_list) == 0:
                        continue
                    event_samples = np.array(event_list)
                    event_times = event_samples / fs
                    
                    # Filter to time range
                    if time_range is not None:
                        mask = (event_times >= time_range[0]) & (event_times <= time_range[1])
                        event_samples_filtered = event_samples[mask].astype(int)
                        event_times_filtered = event_times[mask]
                    else:
                        event_samples_filtered = event_samples.astype(int)
                        event_times_filtered = event_times
                    
                    # Plot vertical lines
                    if len(event_samples_filtered) > 0:
                        for t in event_times_filtered:
                            ax.axvline(t, color=color, linestyle=linestyles[event_type], 
                                     linewidth=1.5, alpha=0.7, label=f'{label} {event_type}')
                        # Remove duplicate labels
                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
    # Plot QRS d2 peaks if provided
    if qrs_d2_peaks_pos is not None:
        for R_pre, R_post in qrs_d2_peaks_pos:
            # Convert to time and filter by time_range
            R_pre_time = R_pre / fs
            R_post_time = R_post / fs
            
            # Check if in time range
            if time_range is not None:
                if R_pre_time >= time_range[0] and R_pre_time <= time_range[1]:
                    ax.axvline(R_pre_time, color='cyan', linestyle=':', 
                             linewidth=1.5, alpha=0.8, label='R_pre')
                if R_post_time >= time_range[0] and R_post_time <= time_range[1]:
                    ax.axvline(R_post_time, color='magenta', linestyle=':', 
                             linewidth=1.5, alpha=0.8, label='R_post')
            else:
                ax.axvline(R_pre_time, color='cyan', linestyle=':', 
                         linewidth=1.5, alpha=0.8, label='R_pre')
                ax.axvline(R_post_time, color='magenta', linestyle=':', 
                         linewidth=1.5, alpha=0.8, label='R_post')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
    
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


def plot_wavelet_details(details, fs, approximation=None, time_range=None, 
                         title_prefix='Wavelet Detail', figsize=(15, 10),
                         p_events=None, qrs_events=None, t_events=None,
                         levels=None):
    """
    Plot wavelet transform detail coefficients at different levels with P-QRS-T events.
    
    Parameters:
    -----------
    details : dict
        Dictionary with keys as levels (1, 2, 3, ...) and values as detail coefficients
    fs : int
        Sampling frequency in Hz
    approximation : numpy array or None
        Optional approximation coefficients to plot as the last subplot
    time_range : tuple or None
        Time range to display as (start, end) in seconds.
        If None, shows the full signal.
    title_prefix : str
        Prefix for subplot titles
    figsize : tuple
        Figure size (width, height)
    p_events : dict or None
        P wave events with keys: 'onset', 'peak', 'offset' (sample indices).
        Will be plotted on detail level 3 (d3)
    qrs_events : dict or None
        QRS events with keys: 'onset', 'peak', 'offset' (sample indices).
        Will be plotted on detail level 2 (d2)
    t_events : dict or None
        T wave events with keys: 'peak', 'offset', 'end' (sample indices).
        Will be plotted on detail level 3 (d3)
    levels : list or None
        List of detail levels to plot (e.g., [2, 3]). If None, plots all levels.
    """
    # Filter details by specified levels
    if levels is not None:
        details_to_plot = {k: v for k, v in details.items() if k in levels}
    else:
        details_to_plot = details
    
    # Determine number of subplots
    n_details = len(details_to_plot)
    n_plots = n_details + (1 if approximation is not None else 0)
    
    # Create subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    
    # Make axes iterable if only one subplot
    if n_plots == 1:
        axes = [axes]
    
    # Get the first detail to create time array
    first_detail = details_to_plot[list(details_to_plot.keys())[0]]
    time = np.arange(len(first_detail)) / fs
    
    # Define colors for different levels
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Track which event types have been added to legend
    legend_labels_added = set()
    
    # Plot detail coefficients
    for idx, level in enumerate(sorted(details_to_plot.keys())):
        ax = axes[idx]
        detail = details[level]
        color = colors[idx % len(colors)]
        
        ax.plot(time, detail, color=color, linewidth=0.8, label=f'd{level}')
        ax.set_ylabel(f'd{level}', fontsize=11, fontweight='bold')
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
        ax.minorticks_on()
        
        # Plot events on appropriate detail levels (only if time_range is specified)
        # QRS events on detail level 2 (d2)
        if level == 2 and qrs_events is not None and time_range is not None:
            event_configs = [(qrs_events, 'QRS', 'purple', {'onset': '--', 'peak': '-', 'offset': '--'})]
            for events, label, event_color, linestyles in event_configs:
                for event_type in linestyles.keys():
                    if event_type in events and events[event_type] is not None:
                        event_list = [e for e in events[event_type] if e is not None]
                        if len(event_list) == 0:
                            continue
                        event_samples = np.array(event_list)
                        event_times = event_samples / fs
                        
                        mask = (event_times >= time_range[0]) & (event_times <= time_range[1])
                        event_times_filtered = event_times[mask]
                        
                        if len(event_times_filtered) > 0:
                            legend_key = f'{label} {event_type}'
                            for i, t in enumerate(event_times_filtered):
                                # Add label only for the first line to avoid duplicates
                                line_label = legend_key if legend_key not in legend_labels_added else None
                                ax.axvline(t, color=event_color, linestyle=linestyles[event_type], 
                                         linewidth=1.5, alpha=0.7, label=line_label)
                                if i == 0 and line_label:
                                    legend_labels_added.add(legend_key)
        
        # P and T events on detail level 3 (d3)
        if level == 3 and (p_events is not None or t_events is not None) and time_range is not None:
            event_configs = [
                (p_events, 'P', 'green', {'onset': '--', 'peak': '-', 'offset': '--'}),
                (t_events, 'T', 'orange', {'peak': '-', 'offset': '--', 'end': ':'})
            ]
            for events, label, event_color, linestyles in event_configs:
                if events is not None:
                    for event_type in linestyles.keys():
                        if event_type in events and events[event_type] is not None:
                            event_list = [e for e in events[event_type] if e is not None]
                            if len(event_list) == 0:
                                continue
                            event_samples = np.array(event_list)
                            event_times = event_samples / fs
                            
                            mask = (event_times >= time_range[0]) & (event_times <= time_range[1])
                            event_times_filtered = event_times[mask]
                            
                            if len(event_times_filtered) > 0:
                                legend_key = f'{label} {event_type}'
                                for i, t in enumerate(event_times_filtered):
                                    # Add label only for the first line to avoid duplicates
                                    line_label = legend_key if legend_key not in legend_labels_added else None
                                    ax.axvline(t, color=event_color, linestyle=linestyles[event_type], 
                                             linewidth=1.5, alpha=0.7, label=line_label)
                                    if i == 0 and line_label:
                                        legend_labels_added.add(legend_key)
        
        ax.legend(loc='upper right', fontsize=9)
        
        # Set time range if specified
        if time_range is not None:
            ax.set_xlim(time_range[0], time_range[1])
    
    # Plot approximation if provided
    if approximation is not None:
        ax = axes[-1]
        time_approx = np.arange(len(approximation)) / fs
        ax.plot(time_approx, approximation, color='black', linewidth=0.8, label='Approximation')
        ax.set_ylabel('Approx', fontsize=11, fontweight='bold')
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
        ax.minorticks_on()
        ax.legend(loc='upper right', fontsize=9)
        
        if time_range is not None:
            ax.set_xlim(time_range[0], time_range[1])
    
    # Set x-label only on the bottom subplot
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    # Set overall title
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.show()
    
    # Print info
    print(f"Number of detail levels: {n_details}")
    print(f"Sampling frequency: {fs} Hz")
    if time_range is not None:
        print(f"Time range: {time_range[0]:.2f}s - {time_range[1]:.2f}s")
    else:
        print(f"Total duration: {len(first_detail)/fs:.1f} seconds")


def plot_ecg_interactive(ecg, fs, annotations=None, time_range=None, title='ECG Signal - Interactive',
                        p_events=None, qrs_events=None, t_events=None):
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
    p_events : dict or None
        P wave events with keys: 'onset', 'peak', 'offset' (sample indices)
    qrs_events : dict or None
        QRS events with keys: 'onset', 'peak', 'offset' (sample indices)
    t_events : dict or None
        T wave events with keys: 'peak', 'offset', 'end' (sample indices)
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
    
    # Plot P-QRS-T events
    event_configs = [
        (p_events, 'P', 'green', {'onset': 'dash', 'peak': 'solid', 'offset': 'dash'}),
        (qrs_events, 'QRS', 'purple', {'onset': 'dash', 'peak': 'solid', 'offset': 'dash'}),
        (t_events, 'T', 'orange', {'peak': 'solid', 'offset': 'dash', 'end': 'dot'})
    ]
    
    for events, label, color, linestyles in event_configs:
        if events is not None:
            for event_type in linestyles.keys():
                if event_type in events and events[event_type] is not None:
                    # Filter out None values
                    event_list = [e for e in events[event_type] if e is not None]
                    if len(event_list) == 0:
                        continue
                    event_samples = np.array(event_list)
                    event_times = event_samples / fs
                    
                    # Filter to time range
                    if time_range is not None:
                        mask = (event_times >= time_range[0]) & (event_times <= time_range[1])
                        event_samples_filtered = event_samples[mask].astype(int)
                        event_times_filtered = event_times[mask]
                    else:
                        event_samples_filtered = event_samples.astype(int)
                        event_times_filtered = event_times
                    
                    # Plot vertical lines
                    if len(event_samples_filtered) > 0:
                        for t in event_times_filtered:
                            fig.add_vline(
                                x=t,
                                line=dict(color=color, width=1.5, dash=linestyles[event_type]),
                                opacity=0.7,
                                annotation_text=f'{label} {event_type}',
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



