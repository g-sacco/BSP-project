# BSP-project

## Source Code Directory (`src/`)

The `src` directory contains the implementation of an ECG waveform delineation algorithm based on the Integer Wavelet Transform (IWT), following the methodology described in Di Marco & Chiari (2011). The implementation includes QRS complex detection and delineation of P and T waves.

### Core Modules

- **`integer_wavelet_transform.py`**: Implements the Integer Wavelet Transform using the "algorithme à trous" (undecimated DWT). Performs multi-level wavelet decomposition with group delay removal based on equations (4.1), (4.2), and (4.3) from the paper.

- **`qrs_detection.py`**: R-peak detection algorithm based on zero-crossing analysis of wavelet detail coefficients at scales 2² and 2³. Implements the Adjacent Modulus Maxima pair (AMMp) method for identifying QRS complexes.

- **`qrs_delineation.py`**: QRS complex delineation at scale 2² to identify QRS onset and offset positions. Uses adaptive thresholds and local maxima search as specified in equation (5.6) of the paper.

- **`p_delineation.py`**: P wave delineation including detection of P peaks, onsets, and offsets. Handles both monophasic and biphasic P waves using conditions (6.1) and (6.2) from the paper.

- **`t_delineation.py`**: T wave delineation to identify T peaks, offsets, and end points. Implements search windows based on RR intervals and QRS offset positions.

### Utility Modules

- **`load_and_preprocess.py`**: Functions for loading ECG records from PhysioNet databases (WFDB format) and resampling to the target frequency (250 Hz). Supports both integer and physical signal values.

- **`ecg_utils.py`**: Helper functions for ECG data loading, processing, and visualization. Includes utilities for working with WFDB records and extracting signal metadata.

- **`results.py`**: Evaluation framework for testing the algorithm on multiple PhysioNet databases (MIT-BIH, QT Database, European ST-T Database). Implements performance metrics including sensitivity, positive predictive value, and delineation errors with configurable tolerance windows.

### Notebooks

- **`main.ipynb`**: Main analysis notebook demonstrating the complete ECG delineation pipeline on sample records.

- **`plot.ipynb`**: Visualization notebook for generating plots and figures of ECG signals and detected/delineated waveforms.

