import database
import analyzer
import random
import datetime

# --- CONFIGURATION ---
BOT_NAME = "Aura"
CRISIS_KEYWORDS = ["suicide", "kill myself", "die", "hurt myself", "end it all", "panic attack"]
GREETINGS = ["hello", "hi", "hey", "good morning", "good evening"]

def get_bot_response(user_input, user_id=1, context_mode=None, bpm=None, stress=None):
    """
    The main brain of Aura.
    Args:
        user_input (str): What the user typed/said.
        context_mode (str): Special triggers like 'post_session' (Automatic Greeting).
        bpm/stress (int): Data from the session if context_mode is active.
    """
    user_input = user_input.lower()

    # ====================================================
    # FEATURE B: AUTO-GREETING (Post-Session Check-In)
    # ====================================================
    if context_mode == "post_session":
        return generate_post_session_analysis(bpm, stress, user_id)

    # ====================================================
    # FEATURE C: CRISIS DETECTION (Safety Protocol)
    # ====================================================
    if any(word in user_input for word in CRISIS_KEYWORDS):
        return (f"⚠️ [EMERGENCY PROTOCOL] I am detecting signs of severe distress. "
                f"Please stop using this app immediately and reach out to a professional "
                f"or call your local emergency services. Your safety is the only priority right now.")

    # ====================================================
    # FEATURE A: LONG-TERM TRENDS (The "Explain My Status" Command)
    # ====================================================
    if "explain" in user_input and ("status" in user_input or "condition" in user_input or "trend" in user_input):
        # We call the Analyzer Module directly
        # We pass dummy current values (0,0) just to get the historical trend
        analysis = analyzer.analyze_session(user_id, 0, 0)
        
        trend = analysis['trend']
        change = analysis['change_pct']
        
        if trend == "Baseline":
            return "I don't have enough history to explain your trends yet. Please record a few more sessions!"
        
        msg = (f"I've analyzed your recent history. Your stress levels have {trend} by {abs(change)}%. "
               f"{analysis['message']}")
        
        if trend == "INCREASED":
            msg += " I strongly suggest checking the 'Games' section to decompress."
        
        return msg

    # ====================================================
    # FEATURE D: GAMIFIED RELIEF (Recommendation Engine)
    # ====================================================
    if "game" in user_input or "bored" in user_input or "fun" in user_input:
        return ("I recommend playing 'Mind Maze' or 'Zen Garden' in our Games section. "
                "Studies show that 5 minutes of focused gaming can lower cortisol levels by 15%.")

    # ====================================================
    # STANDARD CONVERSATION
    # ====================================================
    
    # Greeting
    if any(word in user_input for word in GREETINGS):
        return f"Hello, I am {BOT_NAME}. I am ready to analyze your reports or discuss your mental well-being."

    # Unknown / Out of Scope
    return (f"I specialize in stress analysis and cardiac health. "
            f"For detailed medical diagnoses, please refer to the 'Detailed Description' section in your report "
            f"or consult a doctor. You can ask me to 'Explain my status' or for a 'Relaxation tip'.")

def generate_post_session_analysis(bpm, stress, user_id):
    """
    Special function that runs automatically after a PDF is generated.
    """
    msg = f"Session Complete. I've reviewed your numbers. \n\n"
    msg += f"Your Heart Rate was {bpm} BPM and Stress Level was {stress}%. "
    
    if stress > 75:
        msg += "⚠️ This is high. I detected significant variability in your HRV. "
        msg += "Please do not start work immediately. Shall we try a breathing exercise, or would you like to play a game?"
    elif stress > 40:
        msg += "⚖️ You are in a moderate state. You are capable of working, but keep an eye on your posture."
    else:
        msg += "✅ You are in a state of 'Flow'. Excellent condition."
        
    return msg

def get_bot_response(user_message, user_id=1, context_mode="normal", bpm=0, stress=0):
    user_message = user_message.lower()
    
    # 1. Fetch Latest Report from DB (The "Memory")
    recent_reports = database.get_recent_reports()
    last_report = recent_reports[0] if recent_reports else None
    
    # 2. Logic to handle "How was my session?"
    if "report" in user_message or "stats" in user_message or "last session" in user_message:
        if last_report:
            # Extract data from the dictionary returned by DB
            r_date = last_report['date']
            r_bpm = last_report['avg_bpm']
            r_stress = last_report['avg_stress']
            r_trend = last_report['stress_trend']
            
            return (f"I've pulled up your last report from {r_date}.<br>"
                    f"Your average Heart Rate was <b>{r_bpm} BPM</b> and Stress Level was <b>{r_stress}%</b>.<br>"
                    f"The trend showed: {r_trend}. How are you feeling now?")
        else:
            return "I don't see any saved reports in your history yet. Try running a live session first!"

    # 3. Standard Responses
    if "hello" in user_message or "hi" in user_message:
        return "Hello! I am Aura. I have access to your health history. Ask me about your last session!"
        
    return "I am listening. You can ask me about your stress trends or latest report."