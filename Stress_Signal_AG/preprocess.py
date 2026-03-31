import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION (Based on Brain2Vec Paper) ---
WINDOW_SIZE = 256        # 2 seconds * 128Hz
STEP_SIZE = 128          # 50% overlap (1 second)
# The paper uses 32 EEG channels. 
# In DEAP, channels 0-31 are EEG. 
EEG_CHANNELS = range(32) 

def load_deap_data(file_path):
    """
    Loads a single participant's .dat file (Pickle format).
    Returns: data (trials x channels x samples), labels (trials x 4)
    """
    with open(file_path, 'rb') as f:
        # DEAP data is in Python dictionary format
        data_dict = pickle.load(f, encoding='latin1')
    
    # data_dict['data'] shape: (40 trials, 40 channels, 8064 samples)
    # data_dict['labels'] shape: (40 trials, 4 labels)
    # Labels: Valence, Arousal, Dominance, Liking
    return data_dict['data'], data_dict['labels']

def preprocess_and_segment(data, labels):
    """
    1. Selects only EEG channels.
    2. Segments data into 2-second windows.
    3. Creates binary 'Stress' labels (High Arousal = Stress).
    """
    # 1. Select EEG channels (0-31)
    eeg_data = data[:, EEG_CHANNELS, :] 
    
    segments = []
    stress_labels = []
    
    # Loop through each of the 40 trials (videos)
    for trial_idx in range(eeg_data.shape[0]):
        # Get the signal for this trial (32 channels, 8064 samples)
        trial_signal = eeg_data[trial_idx] 
        
        # Get the 'Arousal' score (Index 1 in DEAP labels)
        arousal_score = labels[trial_idx, 1]
        
        # Define Stress Label: 
        # Paper says: Arousal > 5 is "High Stress" (1), <= 5 is "Low Stress" (0)
        label = 1 if arousal_score > 5 else 0
        
        # 2. Window Segmentation (Sliding Window)
        # 3 seconds pre-trial removed (start from sample 384 as per DEAP docs)
        # But for simplicity, we often just use the whole signal. 
        # Let's slide over the 8064 samples.
        n_samples = trial_signal.shape[1]
        
        for start in range(0, n_samples - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            segment = trial_signal[:, start:end]
            
            # Shape Check: Must be (32, 256)
            if segment.shape[1] == WINDOW_SIZE:
                segments.append(segment)
                stress_labels.append(label)
                
    return np.array(segments), np.array(stress_labels)

def simulate_live_stream(segments):
    """
    Generator that 'yields' one window at a time to mimic a live stream.
    """
    for i, window in enumerate(segments):
        # In a real app, you would add time.sleep(2) here
        yield window

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # REPLACE THIS with your actual path after downloading
    file_path = "s01.dat" 
    
    try:
        print("Loading data...")
        raw_data, raw_labels = load_deap_data(file_path)
        
        print(f"Raw Data Shape: {raw_data.shape}")
        
        print("Preprocessing & Segmenting...")
        X, y = preprocess_and_segment(raw_data, raw_labels)
        
        # Z-Score Normalization (StandardScaler) - Performed on the whole set
        # In a real live stream, you'd normalize using a running mean/std.
        scaler = StandardScaler()
        # Flatten for scaling, then reshape back
        # X shape is (N_segments, 32, 256). We scale across the segments per channel.
        # For simplicity in Phase 1, we can skip complex scaling or just scale the raw values.
        
        print(f"\n✅ Processing Complete!")
        print(f"Total Segments Generated: {X.shape[0]}")
        print(f"Input Shape for Model: {X.shape[1:]} (Channels, Time)")
        print(f"Class Balance: {np.sum(y)} High Stress / {len(y)-np.sum(y)} Low Stress")
        
        print("\n--- Testing Simulator ---")
        stream = simulate_live_stream(X[:5]) # Test first 5 windows
        for idx, window in enumerate(stream):
            print(f"Streaming Window {idx+1}: Shape {window.shape} - Fake Prediction: {'Stress' if idx%2==0 else 'Calm'}")
            
    except FileNotFoundError:
        print("❌ Error: 's01.dat' not found. Please download the dataset first.")