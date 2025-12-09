### convolution "same" no group delay
Validating QRS Detection on MITDB (Resampled to 250Hz)
MITDB Aggregate Results (109734 beats):
Sensitivity (Se): 97.51% (Paper: ~99.78%)
Positive Predictivity (P+): 99.80% (Paper: ~99.86%)

Validating QRS Detection on EDB (Native 250Hz)
EDB Aggregate Results (801999 beats):
Sensitivity (Se): 98.20% (Paper: ~99.81%)
Positive Predictivity (P+): 97.62% (Paper: ~99.56%)

Validating Delineation on QTDB (Native 250Hz)
Fiducial     | Count           | Mean Err (ms)   | Std Dev (ms)   
-----------------------------------------------------------------
P_on         | 2213            | -12.80          | 37.94           (Target: -4.5 ± 13.4)
P_peak       | 2285            | -3.11           | 36.16           (Target: -4.7 ± 9.7)
P_off        | 2292            | -5.88           | 35.07           (Target: -2.5 ± 13.0)
QRS_on       | 3525            | -38.97          | 33.56           (Target: -5.1 ± 7.2)
QRS_off      | 3525            | 43.56           | 39.47           (Target: 0.9 ± 8.7)
T_peak       | 3338            | 9.67            | 43.68           (Target: -0.3 ± 12.8)
T_off        | 3218            | 1.35            | 41.31           (Target: 1.3 ± 18.6)



### NO begin/end conv artifact removal + convolution "full" and group delay
Validating QRS Detection on MITDB (Resampled to 250Hz)

MITDB Aggregate Results (109734 beats):
Sensitivity (Se): 91.27% (Paper: ~99.78%)
Positive Predictivity (P+): 99.79% (Paper: ~99.86%)

Validating QRS Detection on EDB (Native 250Hz)
EDB Aggregate Results (801999 beats):
Sensitivity (Se): 94.03% (Paper: ~99.81%)
Positive Predictivity (P+): 97.51% (Paper: ~99.56%)

Validating Delineation on QTDB (Native 250Hz)
Fiducial     | Count           | Mean Err (ms)   | Std Dev (ms)   
-----------------------------------------------------------------
P_on         | 1573            | 4.02            | 46.81           (Target: -4.5 ± 13.4)
P_peak       | 1650            | 12.36           | 46.92           (Target: -4.7 ± 9.7)
P_off        | 1646            | 8.03            | 45.23           (Target: -2.5 ± 13.0)
QRS_on       | 3170            | -31.29          | 33.66           (Target: -5.1 ± 7.2)
QRS_off      | 3170            | 53.30           | 36.61           (Target: 0.9 ± 8.7)
T_peak       | 2992            | 26.00           | 40.68           (Target: -0.3 ± 12.8)
T_off        | 2923            | 22.17           | 38.40           (Target: 1.3 ± 18.6)