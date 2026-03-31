import numpy as np
import pandas as pd
import pickle
import scipy.signal as signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
FILE_PATH = 'WESAD/S2/S2.pkl' 
FS = 64
WINDOW_SIZE = 60 # 60 seconds per window
STEP_SIZE = 1    # SLIDING WINDOW: Move 1 second at a time (Crucial for more data!)

def load_data(path):
    print("1. Loading Data...")
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    bvp = data['signal']['wrist']['BVP'].flatten()
    labels = data['label']
    
    # Sync Labels
    ratio = len(labels) / len(bvp)
    labels = labels[::int(ratio)][:len(bvp)]
    return bvp, labels

def get_hrv_features(signal_window):
    """
    Extracts standard Heart Rate Variability (HRV) metrics.
    These are what doctors use to detect stress.
    """
    # 1. Peak Detection (find heartbeats)
    peaks, _ = signal.find_peaks(signal_window, distance=30) # Dist 30 samples (~0.5s)
    
    if len(peaks) < 2: return None
    
    # 2. Calculate IBI (Inter-Beat Interval) in ms
    ibi = np.diff(peaks) / FS * 1000
    
    # 3. Extract Features
    mean_hr = 60000 / np.mean(ibi)      # Heart Rate
    std_hr = np.std(ibi)                # SDNN (Variability)
    rmssd = np.sqrt(np.mean(np.diff(ibi)**2)) # RMSSD (Stress Indicator)
    pnn50 = np.sum(np.abs(np.diff(ibi)) > 50) / len(ibi) * 100 # pNN50
    
    return [mean_hr, std_hr, rmssd, pnn50]

def prepare_dataset(bvp, labels):
    print("2. Extracting HRV Features (This creates robust data)...")
    X = []
    y = []
    
    # Use a sliding window with 1-second step to generate THOUSANDS of samples
    # instead of just 64.
    window_len = WINDOW_SIZE * FS
    step = STEP_SIZE * FS 
    
    for i in range(0, len(bvp) - window_len, step):
        window = bvp[i : i + window_len]
        lbl_window = labels[i : i + window_len]
        
        mode_label = np.bincount(lbl_window.astype(int)).argmax()
        
        if mode_label in [1, 2]: # 1=Calm, 2=Stress
            feats = get_hrv_features(window)
            if feats:
                X.append(feats)
                # Map 1->0 (Calm), 2->1 (Stress)
                y.append(0 if mode_label == 1 else 1)
                
    return np.array(X), np.array(y)

# --- MAIN ---
if __name__ == "__main__":
    bvp, labels = load_data(FILE_PATH)
    X, y = prepare_dataset(bvp, labels)
    
    print(f"   -> Dataset Size: {X.shape[0]} samples")
    
    # Stratified Split (Keeps stress/calm ratio balanced)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("3. Training 'Strict' Random Forest...")
    # max_depth=10 prevents the model from memorizing noise (Overfitting fix)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Validation
    preds = model.predict(X_test)
    print("\n--- RESULTS ---")
    print(classification_report(y_test, preds, target_names=['Calm', 'Stress']))
    
    # Save
    with open('stress_model_final.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Final Model Saved as 'stress_model_final.pkl'")