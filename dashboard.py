from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Shared state
camera_frame = None
current_state = "UNKNOWN"
state_history = []
fps = 0
frame_count = 0
fps_start_time = time.time()

lock = threading.Lock()
DB_PATH = "logs/operator_log.db"

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/status")
def status():
    global current_state, fps
    with lock:
        return jsonify({
            "state": current_state,
            "fps": round(fps, 1),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "uptime": get_uptime()
        })

@app.route("/analytics")
def analytics():
    """Advanced analytics using pandas"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Get last 7 days of data
        df = pd.read_sql_query("""
            SELECT timestamp, event 
            FROM logs 
            WHERE datetime(timestamp) >= datetime('now', '-7 days')
        """, conn)
        
        if df.empty:
            return jsonify({"error": "No data"})
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Daily breakdown
        df['date'] = df['timestamp'].dt.date
        daily_stats = df.groupby(['date', 'event']).size().unstack(fill_value=0).to_dict()
        
        # Hourly patterns (24h)
        df['hour'] = df['timestamp'].dt.hour
        hourly_patterns = df.groupby(['hour', 'event']).size().unstack(fill_value=0).to_dict()
        
        # State durations
        state_durations = calculate_state_durations(df)
        
        # Productivity metrics
        total_time = len(df)
        active_time = len(df[df.event == 'ACTIVE'])
        idle_time = len(df[df.event.isin(['IDLE', 'SLEEPING'])])
        absent_time = len(df[df.event == 'ABSENT'])
        
        conn.close()
        
        return jsonify({
            "daily": daily_stats,
            "hourly": hourly_patterns,
            "durations": state_durations,
            "metrics": {
                "total_events": total_time,
                "active_percent": round((active_time / total_time * 100), 1) if total_time > 0 else 0,
                "idle_percent": round((idle_time / total_time * 100), 1) if total_time > 0 else 0,
                "absent_percent": round((absent_time / total_time * 100), 1) if total_time > 0 else 0,
                "active_time": str(timedelta(seconds=active_time)),
                "idle_time": str(timedelta(seconds=idle_time)),
                "absent_time": str(timedelta(seconds=absent_time))
            }
        })
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({"error": str(e)})

def calculate_state_durations(df):
    """Calculate average duration of each state"""
    durations = {}
    
    for state in ['ACTIVE', 'IDLE', 'SLEEPING', 'ABSENT']:
        state_df = df[df.event == state]
        if not state_df.empty:
            # Simple duration estimation
            durations[state] = len(state_df)
    
    return durations

@app.route("/productivity")
def productivity():
    """Get productivity score and trends"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Last 30 days data
        df = pd.read_sql_query("""
            SELECT timestamp, event 
            FROM logs 
            WHERE datetime(timestamp) >= datetime('now', '-30 days')
        """, conn)
        
        if df.empty:
            return jsonify({"score": 0, "trend": "neutral"})
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Calculate daily productivity
        daily_productivity = {}
        for date in df['date'].unique():
            day_df = df[df.date == date]
            active_pct = len(day_df[day_df.event == 'ACTIVE']) / len(day_df) * 100
            daily_productivity[str(date)] = round(active_pct, 1)
        
        # Trend analysis
        scores = list(daily_productivity.values())
        trend = "improving" if len(scores) > 1 and scores[-1] > scores[0] else "declining" if len(scores) > 1 and scores[-1] < scores[0] else "stable"
        
        conn.close()
        
        return jsonify({
            "daily": daily_productivity,
            "trend": trend,
            "current": list(daily_productivity.values())[-1] if daily_productivity else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/export")
def export_data():
    """Export data to CSV"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM logs ORDER BY timestamp DESC", conn)
        conn.close()
        
        # Save to temp file
        temp_file = "temp_export.csv"
        df.to_csv(temp_file, index=False)
        
        return send_file(temp_file, as_attachment=True, download_name=f"operator_logs_{datetime.now().strftime('%Y%m%d')}.csv")
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/video")
def video():
    def generate():
        global camera_frame, fps, frame_count, fps_start_time
        while True:
            with lock:
                if camera_frame is None:
                    continue
                frame = camera_frame.copy()
                
                frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    fps_start_time = time.time()
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps} | State: {current_state}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/log")
def get_logs():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT timestamp, event 
            FROM logs 
            ORDER BY timestamp DESC 
            LIMIT 50
        """, conn)
        conn.close()
        
        return jsonify(df.to_dict('records'))
    except:
        return jsonify([])

def get_uptime():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT timestamp FROM logs ORDER BY timestamp ASC LIMIT 1", conn)
        conn.close()
        
        if not df.empty:
            start_time = pd.to_datetime(df.iloc[0, 0])
            uptime = datetime.now() - start_time
            return str(uptime).split('.')[0]  # Remove microseconds
    except:
        pass
    return "N/A"

def update_state_history(state):
    global state_history
    with lock:
        state_history.append({
            "state": state,
            "time": datetime.now().strftime("%H:%M:%S")
        })
        if len(state_history) > 100:
            state_history = state_history[-100:]

def start_dashboard():
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)