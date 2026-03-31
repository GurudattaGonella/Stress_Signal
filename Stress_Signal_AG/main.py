from datetime import datetime, timedelta  # <--- Essential for your timestamp fix
import os  
import time
import shutil
import collections
import pickle

import cv2
import numpy as np
import scipy.signal as signal
from scipy.signal import find_peaks, butter, filtfilt

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model

# Local Modules
import database
import analyzer
import report_generator

# Initialize DB on startup
database.init_db()
# --- CONFIGURATION ---
MODEL_PATH = "brain2vec_model.h5"     # Your Deep Learning Model (for simulator)
ML_MODEL_PATH = "stress_model_final.pkl" # NEW: Your WESAD Random Forest (for webcam)
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
shared_state = {
    "bpm": 0,
    "stress": 0, # Will now be updated by the ML model
    "is_locked": False,
    "status": "AWAITING FINGER..."
}

class SimulatorState:
    def __init__(self):
        self.current_index = 0
        self.data_segments = None
        self.labels = None
        self.model = None

sim_state = SimulatorState()
real_time_model = None # Placeholder for the new Random Forest

# --- HELPER FUNCTIONS ---
def load_resources():
    global real_time_model
    print("⏳ Loading Models...")
    try:
        # 1. Load Simulator Model (Deep Learning)
        sim_state.model = load_model(MODEL_PATH)
        print("✅ Simulator Model Loaded!")
        
        # 2. Load Real-Time Model (Random Forest) - NEW
        try:
            with open(ML_MODEL_PATH, 'rb') as f:
                real_time_model = pickle.load(f)
            print("✅ Real-Time Stress Model (WESAD) Loaded!")
        except FileNotFoundError:
            print(f"⚠️ WARNING: {ML_MODEL_PATH} not found. Webcam stress will use fallback math.")

        # 3. Load Simulator Data
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
        print(f"⚠️ WARNING: Resource Loading Error ({e}).")

load_resources()

# --- NEW: FEATURE EXTRACTION LOGIC (The "Brain" Bridge) ---
def get_realtime_features(ppg_window, fps=30):
    """
    Extracts [Mean_HR, SDNN, RMSSD, pNN50] from raw PPG buffer.
    Matches the exact logic used in training.
    """
    if len(ppg_window) < fps * 10: return None # Need at least 10s of data

    sig = np.array(ppg_window)
    
    # 1. Filter Noise (0.7Hz - 4Hz)
    try:
        b, a = butter(3, [0.7, 4], btype='bandpass', fs=fps)
        clean_sig = filtfilt(b, a, sig)
    except:
        return None

    # 2. Find Peaks
    # Adjust distance based on FPS (approx 0.5s gap between beats)
    dist = int(fps * 0.5) 
    peaks, _ = find_peaks(clean_sig, distance=dist)
    
    if len(peaks) < 2: return None
    
    # 3. Calculate Metrics
    ibi = np.diff(peaks) / fps * 1000 # Inter-beat intervals in ms
    
    mean_hr = 60000 / np.mean(ibi)
    std_hr = np.std(ibi)            # SDNN
    rmssd = np.sqrt(np.mean(np.diff(ibi)**2)) 
    pnn50 = np.sum(np.abs(np.diff(ibi)) > 50) / len(ibi) * 100
    
    return [mean_hr, std_hr, rmssd, pnn50]

# --- API ENDPOINTS ---
@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/get_live_prediction")
def get_live_prediction():
    if sim_state.current_index >= len(sim_state.data_segments):
        sim_state.current_index = 0
    
    input_data = sim_state.data_segments[sim_state.current_index]
    
    if sim_state.model:
        input_reshaped = input_data.reshape(1, 32, 256)
        prediction = sim_state.model.predict(input_reshaped, verbose=0)
        stress_probability = float(prediction[0][0])
    else:
        stress_probability = 0.5
    
    sim_state.current_index += 1
    is_stressed = stress_probability > 0.35 

    return {
        "stress_score": round(stress_probability * 100, 2),
        "is_high_stress": is_stressed
    }

@app.get("/get_sensor_data")
def get_sensor_data():
    return shared_state

# ==========================================
# MODE B: REAL SENSOR LOGIC
# ==========================================

CAMERA_INDEX = 1 # NOTE: Changed to 0. If external cam, set back to 1.
data_buffer = collections.deque(maxlen=150) # Short buffer for graph (visuals)
bpm_history = collections.deque(maxlen=7)

# --- NEW: STRESS BUFFER ---
# Stores 60 seconds of raw PPG data for the ML model
stress_buffer = collections.deque(maxlen=1800) # 60s * 30fps = 1800

running_min = 0
running_max = 255
stable_frames_count = 0 
REQUIRED_STABLE_FRAMES = 10 
is_signal_locked = False
last_bpm_update_time = 0
current_bpm_display = 0 
current_stress_display = 0 
last_frame_time = 0
current_fps = 30.0 
frames_processed = 0 # Counter to trigger ML prediction

def smooth_data(values, window_size=8):
    if len(values) < window_size: return list(values)
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

# --- HELPER FUNCTION: MEDICAL GRADE BPM MATH ---
# Paste this near the top of main.py, replacing your old BPM function.

def calculate_instant_bpm(values, timestamps, fps):
    """
    Medical-grade BPM calculation using Signal Detrending & Bandpass Filtering.
    Removes 'ghost' trends (baseline drift) and focuses only on true cardiac pulses.
    """
    if len(values) < fps * 1.5: return 0 # Need at least 2 seconds of data
    
    # 1. Convert to numpy for math operations
    sig = np.array(values)
    times = np.array(timestamps)
    
    # 2. DETRENDING (Crucial Fix for the "Wave" Pattern)
    # Removes the slow drift (80->115->50) caused by lighting/breathing
    sig = signal.detrend(sig)
    
    # 3. BANDPASS FILTER (Medical Standard: 0.7Hz - 3.5Hz)
    # Rejects noise below 42 BPM and above 210 BPM
    try:
        b, a = signal.butter(2, [0.7, 4.0], btype='bandpass', fs=fps)
        filtered_sig = signal.filtfilt(b, a, sig)
    except:
        return 0 # Fallback if signal is too short for filter

    # 4. PEAK DETECTION (With Height Threshold)
    # Only count peaks that are distinct (15% of max amplitude)
    # This ignores tiny noise jitters
    amplitude = np.max(filtered_sig) - np.min(filtered_sig)
    min_height = amplitude * 0.10 # Peak must be 15% of total height
    min_dist = int(fps * 0.35) # Refractory period ~450ms (Max 133 BPM)
    
    # We use -filtered_sig because PPG is often inverted (Red absorption)
    peaks, _ = signal.find_peaks(-filtered_sig, distance=min_dist, prominence=min_height)
    
    if len(peaks) < 2: return 0
    
    # 5. CALCULATE BPM
    peak_times = times[peaks]
    time_diffs = np.diff(peak_times)
    
    # Filter impossible beat intervals (Medical limits)
    # 0.33s (180 BPM) to 1.3s (46 BPM)
    valid_diffs = [dt for dt in time_diffs if 0.3 < dt < 1.4]
    
    if len(valid_diffs) < 1: return 0
    
    # Use Median to reject one-off outliers
    avg_beat_time = np.median(valid_diffs)
    bpm = 60.0 / avg_beat_time
    
    return int(bpm)

def calculate_cardiac_stress_fallback(bpm):
    """Fallback if ML model is missing or buffer empty"""
    stress = ((bpm - 55) / 65) * 100
    if stress < 5: stress = 5
    if stress > 95: stress = 95
    return int(stress)

def generate_real_sensor_frames():
    global running_min, running_max, stable_frames_count, is_signal_locked
    global last_bpm_update_time, current_bpm_display, current_stress_display
    global last_frame_time, current_fps, frames_processed
    
    # 1. CAMERA SETUP
    print(f"📷 Attempting to open camera at Index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
    
    if not cap.isOpened():
        print(f"❌ ERROR: Camera at Index {CAMERA_INDEX} failed to open!")
    else:
        print(f"✅ Camera at Index {CAMERA_INDEX} opened successfully!")

    last_frame_time = time.time()
    
    # 2. BUFFERS
    # History buffers for smoothing values over time
    raw_stress_history = collections.deque(maxlen=5) # Increased to 10 for more stability
    bpm_history = collections.deque(maxlen=5)        # Increased to 10
    
    # "Last Valid" memory to prevent drops to 0
    last_valid_bpm = 70
    last_valid_stress = 50
    
    while True:
        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        if dt > 0:
            instant_fps = 1.0 / dt
            current_fps = (0.9 * current_fps) + (0.1 * instant_fps)
            
        success, frame = cap.read()
        if not success: 
            print("❌ Camera read failed.")
            break
            
        h, w, _ = frame.shape
        center_region = frame[h//2-30:h//2+30, w//2-30:w//2+30]
        
        std_dev = 0
        amplitude = 0
        avg_red = 0
        ratio = 0
        
        if center_region.size > 0:
            avg_green = np.mean(center_region[:, :, 1])
            avg_red = np.mean(center_region[:, :, 2])
            if avg_green < 1: avg_green = 1 
            ratio = avg_red / avg_green
            
            # STABILITY FIX 1: Stricter Red Threshold
            # You need a really red finger to count as "Locked"
            if is_signal_locked:
                is_valid = (ratio > 1.3) and (avg_red > 80)
            else:
                is_valid = (ratio > 1.8) and (avg_red > 70) and (avg_red < 255)

            if is_valid:
                stable_frames_count += 1
                data_buffer.append((avg_green, time.time()))
                stress_buffer.append(avg_green)
            else:
                stable_frames_count -= 5 # Decay faster if finger slips
                if stable_frames_count < 0:
                    stable_frames_count = 0
                    data_buffer.clear()
                    data_buffer.append((127, time.time()))
                    stress_buffer.clear()
                    is_signal_locked = False
                    bpm_history.clear() 
                    raw_stress_history.clear()
                    # Don't reset to 0 immediately to avoid flickering UI
                    current_bpm_display = 0 
                    current_stress_display = 0
                    last_valid_bpm = 70
                    last_valid_stress = 50
                    shared_state["status"] = "AWAITING FINGER..."

        show_wave = False
        values = [x[0] for x in data_buffer]
        timestamps = [x[1] for x in data_buffer]
        
        # STABILITY FIX 2: Require more frames before showing data (20 -> 40)
        if len(values) > 50 and stable_frames_count > 20:
            recent_vals = values[-50:]
            local_min = np.min(recent_vals)
            local_max = np.max(recent_vals)
            amplitude = local_max - local_min
            std_dev = np.std(recent_vals)
            
            # STABILITY FIX 3: Noise Rejection
            # If standard deviation is too high, it's movement, not heartbeats.
            if std_dev > 12.0 or amplitude > 80.0:
                 # Don't unlock immediately, just pause updates
                 shared_state["status"] = "HOLD STILL..."
            else:
                if is_signal_locked:
                     show_wave = True
                     shared_state["status"] = "MEASURING..."
                     
                     # --- 1. RUN ML PREDICTION (Background) ---
                     frames_processed += 1
                     if frames_processed % 30 == 0 and len(stress_buffer) > (current_fps * 5):
                         if real_time_model:
                             feats = get_realtime_features(list(stress_buffer), current_fps)
                             if feats:
                                 try:
                                     probs = real_time_model.predict_proba([feats])[0]
                                     raw_score = probs[1] * 100 
                                     raw_stress_history.append(raw_score)
                                 except:
                                     pass

                     # --- 2. UPDATE (Synced & Smoothed) ---
                     if time.time() - last_bpm_update_time > 0.8: # Slower updates (0.8s) = More Stable
                         
                         # A. Calculate Raw BPM
                         clean_vals = smooth_data(values)
                         offset = len(timestamps) - len(clean_vals)
                         valid_timestamps = timestamps[offset:]
                         raw_bpm = calculate_instant_bpm(clean_vals, valid_timestamps, current_fps)
                         
                         if 45 < raw_bpm < 180:
                             # STABILITY FIX 4: Outlier Rejection
                             # If new BPM is way different (>20) from old one, ignore it (unless it's the first reading)

                             #if len(bpm_history) > 0:
                                # prev_bpm = bpm_history[-1]
                                 #if abs(raw_bpm - prev_bpm) > 20:
                                     #raw_bpm = prev_bpm # Reject the jump, keep old value
                             
                             bpm_history.append(raw_bpm)
                             
                             # STABILITY FIX 5: Strong Smoothing (Weighted Average)
                             # New displayed value is 80% old value + 20% new value
                             avg_bpm = int(np.median(bpm_history))
                             last_valid_bpm = avg_bpm
                             current_bpm_display = avg_bpm
                             
                             # B. Update Stress (Hybrid Logic)
                             ml_stress_val = 0
                             if len(raw_stress_history) >= 1:
                                 ml_stress_val = np.median(raw_stress_history)
                             else:
                                 ml_stress_val = calculate_cardiac_stress_fallback(current_bpm_display)

                             # HYBRID RULES
                             bpm_grounding_stress = calculate_cardiac_stress_fallback(current_bpm_display)
                             final_stress = (ml_stress_val * 0.6) + (bpm_grounding_stress * 0.4)
                             
                             if current_bpm_display < 65:
                                 # If HR > 100, minimum stress is 60%
                                 final_stress = min(ml_stress_val, 45)
                             elif current_bpm_display >110:
                                 # If HR < 65, maximum stress is 40%
                                 final_stress = max(ml_stress_val, 60)
                             
                             # Smooth the stress display too
                             last_valid_stress = int((last_valid_stress * 0.9) + (final_stress * 0.1))
                             current_stress_display = last_valid_stress
                         
                         last_bpm_update_time = time.time()

                else:
                     # Check if we should lock
                     is_stable_pulse = (std_dev > 1.5) and (std_dev < 8.0)
                     is_valid_size = (amplitude > 5.0)
                     if is_stable_pulse and is_valid_size: 
                         is_signal_locked = True
                         show_wave = True
                         shared_state["status"] = "ACQUIRING..."
        else:
            shared_state["status"] = "AWAITING FINGER..."

        # Update Shared State
        shared_state["bpm"] = current_bpm_display
        shared_state["stress"] = current_stress_display
        shared_state["is_locked"] = is_signal_locked

        # ... VISUALIZATION BLOCK (Keep your existing one below) ...
        # (This part is purely graphical, so the previous code works fine)
        total_w = 800
        graph_w = 600
        h = 350
        
        dashboard_img = np.zeros((h, total_w, 3), dtype=np.uint8)
        dashboard_img[:] = (20, 15, 10) 
        
        cv2.line(dashboard_img, (graph_w, 0), (graph_w, h), (60, 60, 60), 2)

        grid_spacing = 40
        for x in range(0, graph_w, grid_spacing):
            cv2.line(dashboard_img, (x, 0), (x, h), (40, 35, 30), 1)
        for y in range(0, h, grid_spacing):
            cv2.line(dashboard_img, (0, y), (graph_w, y), (40, 35, 30), 1)

        if len(values) > 15:
            draw_vals = smooth_data(values[-80:])
            if is_signal_locked:
                running_min = (0.9 * running_min) + (0.1 * local_min)
                running_max = (0.9 * running_max) + (0.1 * local_max)
            else:
                running_min = 0
                running_max = 255
            
            if running_max <= running_min + 1: running_max = running_min + 5

            pts = []
            for i, val in enumerate(draw_vals):
                x = int((i / len(draw_vals)) * graph_w)
                if is_signal_locked:
                    norm_val = (val - running_min) / (running_max - running_min)
                    y = int(h - (norm_val * (h * 0.5) + (h * 0.25)))
                else:
                    y = h // 2
                pts.append([x, y])
            
            color = (50, 255, 50) if is_signal_locked else (50, 50, 50)
            cv2.polylines(dashboard_img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)

        # Meter Visualization
        meter_center_x = graph_w + 100
        meter_bottom = h - 80
        meter_top = 80
        max_bar_h = meter_bottom - meter_top
        meter_level = 0
        quality_text = "NO SIGNAL"
        bar_color = (0, 0, 255) 

        is_stable_color = (ratio > 2.0) and (std_dev < 5.0) and (avg_red > 60)
        
        if is_signal_locked or is_stable_color:
            if avg_red > 50:
                meter_level = (avg_red - 50) / 200
                if meter_level > 1.0: meter_level = 1.0
            
            if avg_red > 130:
                bar_color = (0, 255, 0)
                quality_text = "PERFECT"
            else:
                bar_color = (0, 255, 255)
                quality_text = "WEAK"
        else:
            meter_level = 0
            
        bar_h = int(meter_level * max_bar_h)

        cv2.rectangle(dashboard_img, (meter_center_x - 25, meter_top), (meter_center_x + 25, meter_bottom), (50, 50, 50), 2)
        cv2.rectangle(dashboard_img, (meter_center_x - 23, meter_bottom - bar_h), (meter_center_x + 23, meter_bottom), bar_color, -1)
        
        cv2.putText(dashboard_img, "QUALITY", (graph_w + 55, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(dashboard_img, quality_text, (graph_w + 65, meter_bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        ret, buffer = cv2.imencode('.jpg', dashboard_img)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    # ==========================================
#  ADD THIS AT THE END OF MAIN.PY
# ==========================================

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_real_sensor_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
#9th Jan code
@app.get("/test_analysis")
def test_analysis():
    # Simulate a user (ID 1) with current BPM 80 and Stress 75
    # Since DB is empty, it should say "First Session"
    result = analyzer.analyze_session(user_id=1, current_bpm=80, current_stress=75)
    return result
@app.get("/test_pdf")
def test_pdf():
    # 1. Simulate data (Fake history)
    fake_analysis = {
        "message": "Your stress levels have decreased by 10% compared to last week. Keep up the good work with your sleep schedule.",
        "history_dates": ["Mon", "Tue", "Wed", "Thu", "Today"],
        "history_stress": [75, 70, 65, 60, 50]
    }
    # 2. Generate the PDF
    pdf_path = report_generator.generate_pdf("Test User", 72, 50, fake_analysis)
    
    # 3. Send it to the browser to download
    return FileResponse(pdf_path, media_type='application/pdf', filename="Stress_Report.pdf")
import analyzer
import report_generator
import database
from fastapi.responses import FileResponse

@app.get("/stop_session")
def stop_session(user_id: int = 1):
    global current_bpm_display, current_stress_display
    
    print("\n--- 🛑 STOPPING SESSION ---")

    # 1. Capture Data
    final_bpm = current_bpm_display if current_bpm_display > 0 else 75
    final_stress = current_stress_display if current_stress_display > 0 else 50
    print(f"1. Captured Data: BPM={final_bpm}, Stress={final_stress}")

    # 2. Analyze
    try:
        analysis_result = analyzer.analyze_session(user_id, final_bpm, final_stress)
        print("2. Analysis Complete")
    except Exception as e:
        print(f"❌ ERROR in Analysis: {e}")
        return {"status": "error", "message": "Analysis Failed"}
    
    # 3. Save to Database
    try:
        # Save and PRINT if successful
        db_status = database.save_report(
            user_id=user_id,
            avg_bpm=final_bpm,
            avg_stress=final_stress,
            stress_trend=analysis_result['trend'],
            ai_analysis=analysis_result['message']
        )
        print("3. ✅ Database Save: SUCCESS")
    except Exception as e:
        print(f"❌ ERROR in Database Save: {e}")

    # 4. Generate PDF
    try:
        safe_date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"4. Generating PDF for: {safe_date_str}")
        
        pdf_filename = report_generator.generate_pdf(
            user_name="Guest User", 
            bpm=final_bpm, 
            stress=final_stress, 
            analysis_result=analysis_result,
            date=safe_date_str
        )
        
        # --- THE CRITICAL DEBUG PRINTS ---
        # This will print the EXACT full path on your computer
        absolute_path = os.path.abspath(pdf_filename)
        print(f"5. 📄 PDF GENERATED AT: {absolute_path}")
        print(f"   (Please check this specific folder)")
        # ---------------------------------
        
    except Exception as e:
        print(f"❌ ERROR in PDF Generation: {e}")
        return {"status": "error", "message": "PDF Generation Failed"}

    # 5. Return Response
    clean_name = os.path.basename(pdf_filename)
    return {
        "status": "success", 
        "message": "Session Saved", 
        "filename" : clean_name,
        "download_url": f"http://127.0.0.1:8000/download_report?file={clean_name}",
        "analysis": analysis_result
    }
@app.get("/get_recent_reports")
def get_recent_reports():
    """
    Fetches reports and syncs Database Time (UTC) with File Time (Local).
    """
    try:
        raw_reports = database.get_recent_reports()
        
        # Sort Newest First
        # (Assuming returns dicts. If tuples, use x[0])
        raw_reports.sort(key=lambda x: x['id'], reverse=True)

        valid_reports = []

        for rep in raw_reports:
            # 1. Parse the DB Date (UTC)
            # DB Format usually: "2026-02-09 18:34:47"
            db_date_str = rep['date']
            
            try:
                # Convert String to Datetime Object
                utc_time = datetime.strptime(db_date_str, "%Y-%m-%d %H:%M:%S")
                
                # 2. Add 5 Hours 30 Minutes (Convert UTC -> IST)
                local_time = utc_time + timedelta(hours=5, minutes=30)
                
                # 3. Create the Filename expected on disk
                # Format: "2026-02-10_00-04-47"
                safe_local_date = local_time.strftime("%Y-%m-%d_%H-%M-%S")
                expected_filename = f"report_{safe_local_date}.pdf"
                
                # 4. Check if this file exists
                path_in_reports = os.path.join("reports", expected_filename)
                
                if os.path.exists(path_in_reports):
                    # SUCCESS! We found the file.
                    # IMPORTANT: Update the date in the report object to the LOCAL time
                    # so the Frontend generates the correct link.
                    rep['date'] = local_time.strftime("%Y-%m-%d %H:%M:%S")
                    valid_reports.append(rep)
                else:
                    # Fallback: Maybe the file was actually saved in UTC? (Check strictly)
                    # This handles edge cases or older files.
                    safe_utc_date = utc_time.strftime("%Y-%m-%d_%H-%M-%S")
                    utc_filename = f"report_{safe_utc_date}.pdf"
                    if os.path.exists(os.path.join("reports", utc_filename)):
                         valid_reports.append(rep) # Date is already UTC, matches file

            except ValueError:
                # Handle cases where date format might be different
                continue

        return valid_reports

    except Exception as e:
        print(f"Error fetching reports: {e}")
        return []
@app.get("/download_report")
def download_report(file: str):
    """
    Smart Download:
    - Finds the file (Root or Reports folder).
    - Sets 'Content-Disposition' to 'inline'.
    - This forces the browser to OPEN the PDF instead of downloading it.
    """
    # 1. Strip folder prefix
    clean_filename = os.path.basename(file)

    # 2. Find the file path
    path_in_reports = os.path.join("reports", clean_filename)
    
    if os.path.exists(path_in_reports):
        file_path = path_in_reports
    elif os.path.exists(clean_filename):
        file_path = clean_filename
    else:
        return JSONResponse(content={"error": "File not found on server"}, status_code=404)

    # 3. Return with 'inline' header to View instead of Download
    # We manually set the header to "inline" so it opens in a new tab.
    return FileResponse(
        file_path, 
        media_type='application/pdf', 
        headers={"Content-Disposition": f"inline; filename={clean_filename}"}
    )
# ==========================================
#  ADVANCED CHATBOT ROUTES
# ==========================================
import chatbot

@app.get("/chat")
def chat_endpoint(message: str, user_id: int = 1):
    # Standard Chat
    response = chatbot.get_bot_response(message, user_id=user_id)
    return {"reply": response}

@app.get("/chat_trigger")
def chat_trigger(bpm: int, stress: int, user_id: int = 1):
    # Triggers the "Auto-Greeting" after a session
    response = chatbot.get_bot_response("", user_id=user_id, context_mode="post_session", bpm=bpm, stress=stress)
    return {"reply": response}

@app.get("/get_recent_reports")
def get_recent_reports():
    """
    Fetches reports, sorts them (Newest First), and removes any entry 
    where the PDF file is missing (Ghost Entries).
    """
    try:
        # 1. Fetch raw data from DB
        reports = database.get_recent_reports()
        
        # 2. Sort by ID descending (Newest First)
        # We assume the list is a list of dictionaries or tuples. 
        # If it's dictionaries:
        reports.sort(key=lambda x: x['id'], reverse=True)

        valid_reports = []

        # 3. Filter out "Ghost" reports (Missing Files)
        for rep in reports:
            # Construct the expected filename from the date in the DB
            # DB Date format: "2026-02-09 23:49:42"
            # Filename format: "report_2026-02-09_23-49-42.pdf"
            
            date_str = rep['date']
            safe_date = date_str.replace(" ", "_").replace(":", "-")
            expected_filename = f"report_{safe_date}.pdf"
            
            # Check if file exists in 'reports' folder OR root folder
            path_in_reports = os.path.join("reports", expected_filename)
            path_in_root = expected_filename
            
            if os.path.exists(path_in_reports) or os.path.exists(path_in_root):
                valid_reports.append(rep)
            else:
                # Optional: Print which ones are being skipped
                print(f"Skipping missing report: {expected_filename}")

        # 4. Return only the valid, existing reports
        return valid_reports

    except Exception as e:
        print(f"Error fetching reports: {e}")
        return []