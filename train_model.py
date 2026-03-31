import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Reshape, LSTM, Attention
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. DATA LOADING (Same as before) ---
def load_and_process_data(file_path):
    print("Loading s01.dat...")
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    
    raw_data = data_dict['data']
    raw_labels = data_dict['labels']
    
    # Configuration
    WINDOW_SIZE = 256
    STEP_SIZE = 128
    EEG_CHANNELS = range(32)
    
    segments = []
    labels = []
    
    # Select only EEG channels
    eeg_data = raw_data[:, EEG_CHANNELS, :]
    
    for trial_idx in range(eeg_data.shape[0]):
        trial_signal = eeg_data[trial_idx]
        # Label: Arousal > 5 is High Stress (1)
        label = 1 if raw_labels[trial_idx, 1] > 5 else 0
        
        # Sliding Window
        n_samples = trial_signal.shape[1]
        for start in range(0, n_samples - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            segment = trial_signal[:, start:end]
            if segment.shape[1] == WINDOW_SIZE:
                segments.append(segment)
                labels.append(label)
                
    return np.array(segments), np.array(labels)

# --- 2. DEFINE BRAIN2VEC MODEL (Based on Paper Fig. 1) ---
def build_brain2vec_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # The input comes in as (32, 256). Conv2D needs (Height, Width, Channel)
    # We treat EEG Channels (32) as Height, Time (256) as Width, 1 as Depth.
    x = Reshape((32, 256, 1))(inputs)

    # --- CNN Block 1 ---
    # Paper: Conv2D_32 -> BN -> MaxPool
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Shape becomes (16, 128, 32)

    # --- CNN Block 2 ---
    # Paper: Conv2D_64 -> BN -> MaxPool
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Shape becomes (8, 64, 64)

    # --- CNN Block 3 ---
    # Paper: Conv2D_128 -> BN -> MaxPool
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x) # Shape becomes (4, 32, 128)

    # --- Reshape for LSTM ---
    # We need to flatten spatial dims (4, 128) but keep time-like dims (32) for LSTM?
    # Or, following standard Hybrid models: Flatten everything except 'Time'.
    # Paper Fig 1 snippet shows Reshape output (None, 128, 128). 
    # Current shape is (4, 32, 128). 4*32*128 = 16384. 128*128 = 16384.
    # We will reshape to (128, 128) to match the paper's diagram exactly.
    x = Reshape((128, 128))(x)

    # --- LSTM Layer ---
    # Paper: LSTM Output (None, 128, 64)
    # return_sequences=True is needed for Attention
    x_lstm = LSTM(64, return_sequences=True)(x)

    # --- Attention Mechanism ---
    # Paper: Attention Output (None, 128, 64)
    # Self-attention: Query and Value are both the LSTM output
    x = Attention()([x_lstm, x_lstm])

    # --- Classification Head ---
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x) # Binary Classification (Stress vs Calm)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- 3. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    # A. Load Data
    X, y = load_and_process_data("s01.dat")
    
    # B. Split Train/Test (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")
    
    # C. Build Model
    # Input shape is (32, 256) because we removed the extra dimension during load
    model = build_brain2vec_model(input_shape=(32, 256))
    model.summary()
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    # D. Train
    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        callbacks=[early_stop],          # 20 Epochs is usually enough for s01.dat
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # E. Save Model
    model.save("brain2vec_model.h5")
    print("\n✅ Model Saved as 'brain2vec_model.h5'")