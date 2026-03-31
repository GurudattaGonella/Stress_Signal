from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import collections
import time
from fastapi.responses import StreamingResponse

# --- CONFIGURATION ---
MODEL_PATH = "brain2vec_model.h5"
DATA_PATH = "s01.dat"
EEG_CHANNELS = range(32)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
class SimulatorState:
    def __init__(self):
        self.current_index = 0
        self.data_segments = None
        self.labels = None
        self.model = None

sim_state = SimulatorState()

# --- HELPER FUNCTIONS ---
def load_resources():
    print("⏳ Loading Brain2Vec Model...")
    try:
        sim_state.model = load_model(MODEL_PATH)
        print("✅ Model Loaded!")
        
        print(f"⏳ Loading Data from {DATA_PATH}...")
        with open(DATA_PATH, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        
        raw_data = data_dict['data'] 
        raw_labels = data_dict['labels']
        eeg_data = raw_data[:, EEG_CHANNELS, :]
        
        segments = []
        labels = []
        WINDOW_SIZE = 256
        STEP_SIZE = 128
        
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
        
        sim_state.data_segments = np.array(segments)
        sim_state.labels = np.array(labels)
        print(f"✅ Data Loaded! Ready to stream {len(segments)} windows.")
    except Exception as e:
        print(f"⚠️ WARNING: Could not load model/data ({e}).")

load_resources()

# --- API ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "Brain2Vec API is Running"}

@app.get("/start_simulation")
def start_simulation():
    sim_state.current_index = 0
    return {"message": "Simulation started", "total_steps": len(sim_state.data_segments)}

@app.get("/get_live_prediction")
def get_live_prediction():
    if sim_state.current_index >= len(sim_state.data_segments):
        sim_state.current_index = 0
    
    input_data = sim_state.data_segments[sim_state.current_index]
    actual_label = int(sim_state.labels[sim_state.current_index])
    
    if sim_state.model:
        input_reshaped = input_data.reshape(1, 32, 256)
        prediction = sim_state.model.predict(input_reshaped, verbose=0)
        stress_probability = float(prediction[0][0])
    else:
        stress_probability = 0.5
    
    sim_state.current_index += 1
    is_stressed = stress_probability > 0.35 

    return {
        "timestamp_index": sim_state.current_index,
        "stress_score": round(stress_probability * 100, 2),
        "is_high_stress": is_stressed,
        "actual_label": "High Stress" if actual_label == 1 else "Low Stress"
    }

@app.get("/manual_assessment")
def manual_assessment(sleep_hours: int, anxiety_level: int):
    score = 0
    if sleep_hours < 6: score += 40
    elif sleep_hours < 8: score += 20
    score += (anxiety_level * 6)
    final_score = min(score, 100)
    return {
        "stress_score": final_score,
        "is_high_stress": final_score > 50,
        "message": "Calculated based on self-report"
    }

# ==========================================
# MODE B: REAL SENSOR (RATIO LOCK)
# ==========================================

CAMERA_INDEX = 1 
data_points = collections.deque(maxlen=100)
running_min = 0
running_max = 255

# Stability Logic
stable_frames_count = 0 
REQUIRED_STABLE_FRAMES = 12 

def smooth_data(data, window_size=8):
    if len(data) < window_size:
        return list(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def generate_real_sensor_frames():
    global running_min, running_max, stable_frames_count
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    for _ in range(10): cap.read()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        h, w, _ = frame.shape
        box_size = 40
        center_region = frame[h//2-box_size:h//2+box_size, w//2-box_size:w//2+box_size]
        
        finger_detected = False
        ratio = 0.0
        
        if center_region.size > 0:
            avg_red = np.mean(center_region[:, :, 2]) 
            avg_green = np.mean(center_region[:, :, 1])
            
            # --- THE MAGIC RATIO CALCULATION ---
            # Avoid division by zero
            if avg_green < 1: avg_green = 1 
            
            ratio = avg_red / avg_green
            
            # --- STRICT RULES ---
            # 1. BRIGHTNESS: Red must be decent (> 100)
            is_bright = avg_red > 100
            
            # 2. BLOOD ABSORPTION CHECK (The Wall Killer)
            # A wall usually has a ratio of 1.0 to 1.8.
            # A finger usually has a ratio > 3.0 (Red is 3x higher than Green).
            # We set the threshold at 2.5 to be safe.
            is_biological = ratio > 2.5 

            if is_bright and is_biological:
                stable_frames_count += 1
                val = avg_green 
                data_points.append(val)
            else:
                # FAILED
                stable_frames_count = 0
                if len(data_points) > 0:
                    data_points.append(data_points[-1])
                else:
                    data_points.append(127)

        show_wave = stable_frames_count > REQUIRED_STABLE_FRAMES

        # --- DRAWING ---
        scope_h, scope_w = 300, 600
        scope_img = np.zeros((scope_h, scope_w, 3), dtype=np.uint8)
        scope_img[:] = (42, 23, 15) 

        if len(data_points) > 20:
            clean_data = smooth_data(list(data_points))
            
            if show_wave:
                curr_min = min(clean_data)
                curr_max = max(clean_data)
                running_min = (0.9 * running_min) + (0.1 * curr_min)
                running_max = (0.9 * running_max) + (0.1 * curr_max)
                if running_max <= running_min + 1:
                    running_max = running_min + 5
            else:
                running_min = 0
                running_max = 255

            pts = []
            for i, val in enumerate(clean_data):
                x = int((i / len(clean_data)) * scope_w)
                if show_wave:
                    norm_val = (val - running_min) / (running_max - running_min)
                    y = int(scope_h - (norm_val * (scope_h * 0.7) + (scope_h * 0.15)))
                else:
                    y = scope_h // 2
                pts.append([x, y])
            
            points_array = np.array(pts, np.int32).reshape((-1, 1, 2))
            color = (0, 255, 200) if show_wave else (50, 50, 50)
            cv2.polylines(scope_img, [points_array], isClosed=False, color=color, thickness=4, lineType=cv2.LINE_AA)

        # --- DEBUG TEXT (Helps you see why the wall is failing) ---
        # ratio:.1f means show 1 decimal place (e.g., "1.5")
        #color_ratio = (0, 255, 0) if ratio > 2.5 else (0, 0, 255)
        #info_text = f"COLOR RATIO: {ratio:.1f} (Must be > 2.5)"
        #cv2.putText(scope_img, info_text, (300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ratio, 1)
        
        if show_wave:
             cv2.putText(scope_img, "SIGNAL LOCKED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
             if stable_frames_count > 0:
                 cv2.putText(scope_img, "VERIFYING...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
             else:
                 cv2.putText(scope_img, "AWAITING BIOMETRIC SIGNAL...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        ret, buffer = cv2.imencode('.jpg', scope_img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_real_sensor_frames(), media_type="multipart/x-mixed-replace; boundary=frame")