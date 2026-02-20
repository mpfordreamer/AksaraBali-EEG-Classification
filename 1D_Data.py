# IMPORT LIBRARIES
import os
import sys
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter, filtfilt
import glob

# BASIC UTILITY FUNCTIONS
def read_file(file):
    """Load MATLAB file"""
    data = sio.loadmat(file)
    return data

def butter_bandpass_filter(x, lowcut, highcut, fs, order=4):
    """Zero-phase Butterworth bandpass."""
    nyq = 0.5 * fs
    low  = max(lowcut/nyq, 0.001)
    high = min(highcut/nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)


def compute_DE(signal):
    """Compute Differential Entropy"""
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2

def compute_DE_per_band(signal, frequency, feature_func):
    """
    Compute DE for each band (theta, alpha, beta, gamma) for a given 1D signal.
    Returns list: [DE_theta, DE_alpha, DE_beta, DE_gamma]
    """
    theta = butter_bandpass_filter(signal, 4, 8, frequency, order=4)
    alpha = butter_bandpass_filter(signal, 8, 14, frequency, order=4)
    beta  = butter_bandpass_filter(signal, 14, 31, frequency, order=4)
    gamma = butter_bandpass_filter(signal, 31, 45, frequency, order=4)
    return [
        feature_func(theta),
        feature_func(alpha),
        feature_func(beta),
        feature_func(gamma)
    ]

# MAIN PROCESSING FUNCTION
def decompose_band_based_relative(data, feature_func=compute_DE):
    frequency = 100
    n_trials = 12
    n_channels = 16
    eps = 1e-8  

    all_trial_features = []
    all_base_features = []
    data_seconds_list = []

    for trial in range(n_trials):
        trial_data = data[:, trial][0]  # (N, 16)
        n_time = trial_data.shape[0]
        data_seconds = n_time // frequency
        data_seconds_list.append(data_seconds)

        # Baseline: first 300 samples
        baseline_data = trial_data[:300, :]  # (300, 16)
        baseline_DE = []
        for ch in range(n_channels):
            de_bands = compute_DE_per_band(baseline_data[:, ch], frequency, feature_func)
            baseline_DE.append(de_bands)
        baseline_DE = np.array(baseline_DE)  # (16, 4)
        all_base_features.append(baseline_DE.flatten())

        # Trial: per second, per channel, per band
        trial_DE = []
        for ch in range(n_channels):
            ch_DE = []
            for sec in range(data_seconds):
                segment = trial_data[sec * frequency:(sec + 1) * frequency, ch]
                de_bands = compute_DE_per_band(segment, frequency, feature_func)
                ch_DE.append(de_bands)  # (4,)
            ch_DE = np.array(ch_DE)  # (data_seconds, 4)
            trial_DE.append(ch_DE)
        trial_DE = np.array(trial_DE)  # (16, data_seconds, 4)

        # Relative difference: y_ij(t) = f_ij(t) / f_ij(0)
        trial_DE_relative = trial_DE / (baseline_DE[:, np.newaxis, :] + eps)  # broadcasting

        # Reshape to (data_seconds, 64)
        trial_DE_relative = np.transpose(trial_DE_relative, (1, 0, 2)).reshape(data_seconds, -1)
        all_trial_features.append(trial_DE_relative)

    base_features = np.array(all_base_features)  # (12, 64)
    trial_features = np.vstack(all_trial_features)  # (12*data_seconds, 64)
    data_seconds_list = np.array(data_seconds_list)
    return base_features, trial_features, data_seconds_list

# LABEL EXTRACTION
def get_labels(data_labels, data_trial):
    """Extract labels and expand to per-second length for each trial"""
    final_valence_labels = np.empty([0])
    final_arousal_labels = np.empty([0])
    final_dominance_labels = np.empty([0])
    for trial in range(12):
        data_label = data_labels[:, trial][0][0].copy()
        data_trial_row = (data_trial[:, trial][0].shape[0] // 100)

        # valence, arousal, dominance (boolean per second)
        valence_labels   = np.array([data_label[0] == 1] * data_trial_row)
        arousal_labels   = np.array([data_label[1] == 1] * data_trial_row)
        dominance_labels = np.array([data_label[2] == 1] * data_trial_row)

        final_valence_labels   = np.append(final_valence_labels, valence_labels)
        final_arousal_labels   = np.append(final_arousal_labels, arousal_labels)
        final_dominance_labels = np.append(final_dominance_labels, dominance_labels)

    return final_arousal_labels, final_valence_labels, final_dominance_labels

# BASELINE REDUCTION
def baseline_reduction(joined_data, signal_clean):
    assert signal_clean.shape == (16, 300), "signal_clean must have shape (16, 300)"

    # Baseline stats per channel
    baseline_mean = np.mean(signal_clean, axis=1)  # (16,)
    baseline_std  = np.std(signal_clean, axis=1)   # (16,)

    for trial_idx in range(len(joined_data)):
        trial_signal = joined_data[trial_idx][0]  # (N, 16)
        n_time = trial_signal.shape[0]

        # Repeat baseline stats along time
        baseline_mean_repeated = np.tile(baseline_mean, (n_time, 1))  # (N, 16)
        baseline_std_repeated  = np.tile(baseline_std, (n_time, 1))   # (N, 16)

        # Avoid division by zero
        baseline_std_repeated[baseline_std_repeated < 1e-8] = 1.0

        # Z-score normalization
        trial_normalized = (trial_signal - baseline_mean_repeated) / baseline_std_repeated

        # Build repeated baseline to match trial length
        signal_clean_T = signal_clean.T  # (300, 16)
        n_repeats = n_time // 300 + 1
        baseline_repeated = np.tile(signal_clean_T, (n_repeats, 1))[:n_time, :]  # (N, 16)

        # Baseline reduction after normalization
        reduced_trial = trial_normalized - ((baseline_repeated - baseline_mean_repeated) / baseline_std_repeated)

        # Write back
        joined_data[trial_idx][0] = reduced_trial

    return joined_data

# MAIN EXECUTION
if __name__ == '__main__':
    # Mode selection: "with" (apply baseline reduction) or "without" (skip)
    # Usage example: python script.py with
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "with"
    USE_BASELINE = (mode == "with")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Define dataset paths
    input_dir    = os.path.join(BASE_DIR, "datasets", "raw", "CS_Train")  # Standardized input
    baseline_dir = os.path.join(BASE_DIR, "datasets", "baseline")         # Standardized baseline

    # Define output path depending on mode
    output_root = os.path.join(BASE_DIR, "datasets")
    output_dir = os.path.join(BASE_DIR, "datasets", "features", "CS_Train") # Standardized output
    os.makedirs(output_dir, exist_ok=True)

    # Collect files
    input_files = sorted(glob.glob(os.path.join(input_dir, '*.mat')))
    baseline_files = sorted(glob.glob(os.path.join(baseline_dir, '*.mat'))) if USE_BASELINE else []

    if USE_BASELINE and (len(baseline_files) != len(input_files)):
        print("[WARN] Number of baseline files does not match input files. Zipping will stop at shortest list.")

    # Iterate through file pairs (or inputs only if without baseline)
    iterable = zip(input_files, baseline_files) if USE_BASELINE else [(f, None) for f in input_files]

    for input_file, baseline_file in iterable:
        filename = os.path.basename(input_file)
        print(f"Processing {filename}...  (mode: {'with baseline' if USE_BASELINE else 'without baseline'})")

        # Load input file
        input_data = read_file(input_file)
        joined_data = input_data['joined_data'].copy()  # (12, 1) cell array

        # Optional: apply baseline reduction using external baseline file
        if USE_BASELINE:
            baseline_data = read_file(baseline_file)
            signal_clean = baseline_data['signal_clean']  # (16, 300)
            joined_data_processed = baseline_reduction(joined_data, signal_clean)
        else:
            joined_data_processed = joined_data  # no baseline reduction

        # Feature extraction (with relative difference inside)
        base_features, trial_features, data_seconds_list = decompose_band_based_relative(
            joined_data_processed,
            feature_func=compute_DE
        )

        # Labels expanded to per-second
        arousal_labels, valence_labels, dominance_labels = get_labels(
            input_data['labels_selfassessment'].copy(),
            joined_data_processed
        )

        # Save output .mat
        output_file = os.path.join(output_dir, f"DE_{filename}")
        sio.savemat(
            output_file,
            {
                "data": trial_features,
                "valence_labels": valence_labels,
                "arousal_labels": arousal_labels,
                "dominance_labels": dominance_labels
            }
        )
        print(f"Saved: {output_file}")