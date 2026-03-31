import random

def analyze_session(user_id, avg_bpm, avg_stress):
    """
    Analyzes the user's session data and returns simple, friendly advice.
    """
    
    # 1. Determine the Trend (Simulated)
    trends = ["Stable", "Improving", "Elevated"]
    trend = random.choice(trends)

    # 2. Friendly Logic
    if avg_stress < 40:
        # Relaxed
        message = (
            "You are doing great! Your stress levels are in the 'Optimal Zone.' "
            "This means your body is relaxed and balanced. "
            "Whatever you are doing right now is working perfectly for you."
        )
        recommendation = "Keep it up! You are calm and focused."
        status = "OPTIMAL"

    elif 40 <= avg_stress < 70:
        # Normal
        message = (
            "You are doing fine. Your stress levels are normal for a typical day. "
            "You might feel a little busy or focused, but your body is handling it well. "
            "No need to worry—just keep going!"
        )
        recommendation = "Take a quick deep breath if you feel busy."
        status = "NORMAL"

    else:
        # High
        message = (
            "You seem a bit stressed right now. "
            "Your sensors picked up some higher tension levels. "
            "It happens to everyone! It just means your body needs a quick timeout to reset."
        )
        recommendation = "Let's take a 2-minute break to relax."
        status = "ELEVATED"

    # 3. Return the Data Package
    return {
        "trend": trend,
        "message": message,
        "recommendation": recommendation,
        "status": status
    }