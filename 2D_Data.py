import time
from sklearn import preprocessing
import scipy.io as sio
import numpy as np
import os
import glob

def read_file(file):
    """Load processed DE feature data"""
    file = sio.loadmat(file)
    trial_data = file['data']  
    base_data = file["base_data"] 
    data_seconds_list = file['data_seconds_list']
    
    if data_seconds_list.ndim > 1:
        data_seconds_list = data_seconds_list.flatten()
    
    return trial_data, base_data, file["valence_labels"], file["arousal_labels"], file['dominance_labels'], data_seconds_list

def process_data_for_lstm_2d(path, use_baseline=True):
    """Process data for LSTM: output shape (total_timesteps, features)"""
    trial_data, base_data, valence_labels, arousal_labels, dominance_labels, data_seconds_list = read_file(path)
    
    # Ensure shape (12, timesteps_per_trial, features)
    if trial_data.shape == (60, 64):
        sequences = trial_data.reshape(12, 5, 64)  # 12 trials, 5 seconds, 64 features
    else:
        timesteps_per_trial = trial_data.shape[0] // 12
        sequences = trial_data.reshape(12, timesteps_per_trial, trial_data.shape[1])
    
    # Flatten to 2D (total_timesteps, features)
    data = sequences.reshape(-1, sequences.shape[-1])
    
    print("final data shape:", data.shape)
    return data, valence_labels, arousal_labels, dominance_labels

# ==============================================================
# MAIN EXECUTION
# ==============================================================

if __name__ == '__main__':
    # Define base directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Dataset directory selection
    # CS: Character Spelling data
    # CI: Character Imagined data
    dataset_dir = os.path.join(BASE_DIR, "datasets", "features", "CS_Train")  # Standardized input from 1D_Data
    use_baseline = 'yes'
    
    # LSTM 2D processing
    output_dir = os.path.join(BASE_DIR, "datasets", "train", "CS_Train")  # Standardized output for TrainLSTM
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing files from: {dataset_dir}")
    print(f"Saving results to: {output_dir}")
    
    for file in sorted(glob.glob(os.path.join(dataset_dir, '*.mat'))):
        filename = os.path.basename(file)
        print(f"Processing for LSTM 2D: {filename}")
        try:
            data, valence_labels, arousal_labels, dominance_labels = process_data_for_lstm_2d(
                file,
                use_baseline=(use_baseline == "yes")
            )
            output_path = os.path.join(output_dir, filename)
            sio.savemat(output_path, {
                "data": data,
                "valence_labels": valence_labels,
                "arousal_labels": arousal_labels,
                "dominance_labels": dominance_labels
            })
            print(f"  Saved for LSTM 2D: {data.shape}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    print("Processing completed!")
