import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# CONFIGURATION
# Update this path once you unzip the file
file_path = 'WESAD/S2/S2.pkl' 

def load_wesad(path):
    print(f"Loading {path}...")
    with open(path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data

try:
    # 1. Load the data
    data = load_wesad(file_path)
    
    # 2. Extract Wrist BVP (Blood Volume Pulse) - This matches your Webcam PPG
    # The structure is data['signal']['wrist']['BVP']
    bvp_signal = data['signal']['wrist']['BVP']
    
    # 3. Extract Labels (0=Baseline, 1=Stress, 2=Amusement)
    labels = data['label']
    
    print("\n✅ Data Loaded Successfully!")
    print(f"BVP Signal Shape: {bvp_signal.shape}")
    print(f"Labels Shape: {labels.shape}")
    
    # 4. Visualization (To confirm it looks like your webcam data)
    plt.figure(figsize=(12, 4))
    plt.plot(bvp_signal[:1000], label='WESAD BVP (Wrist)', color='green') # Plot first 1000 points
    plt.title("Sample BVP Signal from WESAD (Subject 2)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("❌ File not found. Make sure you extracted the zip and updated 'file_path'.")