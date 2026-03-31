import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. DATA LOADING FUNCTION (Same as training) ---
def load_and_process_data(file_path):
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    
    raw_data = data_dict['data']
    raw_labels = data_dict['labels']
    
    WINDOW_SIZE = 256
    STEP_SIZE = 128
    EEG_CHANNELS = range(32)
    
    segments = []
    labels = []
    
    eeg_data = raw_data[:, EEG_CHANNELS, :]
    
    for trial_idx in range(eeg_data.shape[0]):
        trial_signal = eeg_data[trial_idx]
        label = 1 if raw_labels[trial_idx, 1] > 5 else 0
        
        n_samples = trial_signal.shape[1]
        for start in range(0, n_samples - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            segment = trial_signal[:, start:end]
            if segment.shape[1] == WINDOW_SIZE:
                segments.append(segment)
                labels.append(label)
                
    return np.array(segments), np.array(labels)

# --- 2. MAIN EVALUATION ---
if __name__ == "__main__":
    # A. Load Data
    X, y = load_and_process_data("s01.dat")
    
    # B. Re-create the exact same split used in training
    # random_state=42 ensures we get the EXACT same 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nEvaluating on {len(X_test)} Test Samples...")
    
    # C. Load the Trained Model
    model = load_model("brain2vec_model.h5")
    
    # D. Get Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.35).astype(int) # Convert probabilities to 0 or 1
    
    # E. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("FINAL MODEL RESULTS (Week 2 Milestone)")
    print("="*40)
    print(f"✅ Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"✅ Precision: {prec:.4f}  ({prec*100:.2f}%)")
    print(f"✅ Recall:    {rec:.4f}  ({rec*100:.2f}%)")
    print(f"✅ F1-Score:  {f1:.4f}   ({f1*100:.2f}%)")
    print("="*40)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Low Stress', 'High Stress']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)