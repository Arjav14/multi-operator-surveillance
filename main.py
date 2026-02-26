import cv2
import time
import yaml
import numpy as np
import threading
from flask import Flask, render_template, Response, jsonify, request
from detector import HumanDetector
from db_logger import Logger
from alert import Alert
from multi_operator import MultiOperatorMonitor
from datetime import datetime
import traceback
from collections import deque

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Global variables
latest_frame = None
current_state = "UNKNOWN"
state_history = []
frame_lock = threading.Lock()
state_lock = threading.Lock()
start_time = time.time()
state_start_time = time.time()
no_operator_warning_shown = False

# Notifications queue
notifications = deque(maxlen=50)

# Multi-operator monitor
operator_monitor = None

# Flask app
app = Flask(__name__)

# ============= NOTIFICATION TRACKING =============
# Track last notification sent per operator per state
# This ensures we only send ONE notification per state change
operator_last_notification = {}  # Format: {operator_id: {'state': last_state, 'time': timestamp}}

def should_send_notification(operator_id, new_state):
    """Check if we should send notification based on state change"""
    global operator_last_notification
    
    current_time = time.time()
    
    # If operator never had notification, send it
    if operator_id not in operator_last_notification:
        operator_last_notification[operator_id] = {
            'state': new_state,
            'time': current_time
        }
        return True
    
    last_data = operator_last_notification[operator_id]
    last_state = last_data['state']
    
    # Only send if state has CHANGED
    if last_state != new_state:
        # Update tracking
        operator_last_notification[operator_id] = {
            'state': new_state,
            'time': current_time
        }
        return True
    
    # Same state, don't send notification
    return False

def add_notification(message, type="info", operator_id=None, state=None):
    """Add notification to dashboard queue based on state change"""
    
    # If this is an operator-specific notification, check if we should send it
    if operator_id is not None and state is not None:
        if not should_send_notification(operator_id, state):
            return  # Don't send duplicate notification for same state
    
    notification = {
        'id': len(notifications) + 1,
        'message': message,
        'type': type,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'read': False,
        'operator': operator_id,
        'state': state
    }
    notifications.appendleft(notification)
    print(f"[{type.upper()}] {message}")

def reset_operator_notifications(operator_id):
    """Force reset notifications for an operator (used when they leave)"""
    global operator_last_notification
    if operator_id in operator_last_notification:
        del operator_last_notification[operator_id]
# ==================================================

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global latest_frame
        while True:
            try:
                with frame_lock:
                    if latest_frame is None:
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, "Waiting for camera...", (150, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        frame_to_send = blank
                    else:
                        frame_to_send = latest_frame.copy()
                
                _, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
            except Exception as e:
                print(f"❌ Streaming error: {e}")
                continue
            
            time.sleep(0.03)
    
    return Response(generate(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    """Get current system state"""
    try:
        with state_lock:
            duration = int(time.time() - state_start_time)
            return jsonify({
                'state': current_state,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'uptime': int(time.time() - start_time),
                'duration': duration
            })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/operators')
def get_operators():
    """Get current operator status"""
    try:
        global operator_monitor
        if operator_monitor:
            return jsonify(operator_monitor.get_summary())
        return jsonify({'occupied_slots': 0, 'active_count': 0, 'idle_count': 0, 'operators': []})
    except Exception as e:
        print(f"❌ Operators API error: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/notifications')
def get_notifications():
    """Get recent notifications for dashboard"""
    try:
        return jsonify(list(notifications))
    except Exception as e:
        return jsonify([])

@app.route('/api/notifications/mark-read', methods=['POST'])
def mark_notifications_read():
    """Mark all notifications as read"""
    try:
        for notification in notifications:
            notification['read'] = True
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/operator_history')
def get_operator_history():
    """Get operator history"""
    try:
        global operator_monitor
        if operator_monitor:
            slot = request.args.get('slot', type=int)
            if slot:
                slot -= 1
            return jsonify(operator_monitor.get_history(slot))
        return jsonify([])
    except Exception as e:
        return jsonify([])

def run_dashboard():
    """Run Flask app in separate thread"""
    try:
        print("📊 Starting dashboard server...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

def draw_bounding_boxes(frame, active_operators):
    """Draw bounding boxes on frame"""
    for op in active_operators:
        x1, y1, x2, y2 = op['box']
        
        if op['state'] == 'ACTIVE':
            color = (0, 255, 0)
            label = f"#{op['slot']}"
        else:
            color = (0, 255, 255)
            label = f"#{op['slot']} IDLE ({op['idle_duration']}s)"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1-25), (x1+w, y1-5), (0,0,0), -1)
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return frame

def main():
    global latest_frame, current_state, state_history, start_time, state_start_time
    global operator_monitor, no_operator_warning_shown
    
    print("="*60)
    print("🚀 Multi-Operator Monitoring System")
    print("="*60)
    
    # Start dashboard
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    time.sleep(2)
    
    print(f"📊 Dashboard: http://localhost:5000")
    
    # Initialize camera
    print("📹 Initializing camera...")
    cap = cv2.VideoCapture(config['camera']['source'], cv2.CAP_DSHOW)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Camera failed to open with DSHOW, trying default...")
        cap = cv2.VideoCapture(config['camera']['source'])
    
    if not cap.isOpened():
        print("❌ Camera failed to open")
        return
    
    print("✅ Camera initialized successfully")
    
    # Initialize components
    print("🤖 Loading YOLO model...")
    detector = HumanDetector()
    logger = Logger(config['logging']['csv_path'])
    alert = Alert()
    
    # Initialize multi-operator monitor
    operator_monitor = MultiOperatorMonitor(
        max_operators=4,
        idle_threshold=config['thresholds']['idle_seconds'],
        absence_threshold=config['thresholds']['absence_seconds']
    )
    
    # Thresholds
    IDLE_TIME = config['thresholds']['idle_seconds']
    ABSENT_TIME = config['thresholds']['absence_seconds']
    
    # Motion filtering settings
    MOTION_THRESHOLD = 15000
    MIN_MOTION_AREA = 500
    BLUR_SIZE = 5
    
    print(f"⚙️ Motion Threshold: {MOTION_THRESHOLD}")
    print(f"⚙️ Idle Threshold: {IDLE_TIME}s")
    print(f"⚙️ Absent Threshold: {ABSENT_TIME}s")
    print(f"⚙️ Max Operators: 4")
    print("✅ System running. Press Ctrl+C to stop.")
    print("-" * 60)
    
    # State tracking
    last_seen = time.time()
    prev_gray = None
    prev_state = None
    
    frame_count = 0
    state = "UNKNOWN"
    fps = 0
    fps_counter = 0
    fps_time = time.time()
    no_operator_start_time = None
    
    # Track active operators for join/leave detection
    previous_operators = set()
    
    # Add startup notification
    add_notification("🚀 System started successfully", "success")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame, reconnecting...")
                add_notification("⚠️ Camera connection lost - reconnecting...", "warning")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(config['camera']['source'], cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['height'])
                continue
            
            current_time = time.time()
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS
            if current_time - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = current_time
            
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
            
            # Detect multiple operators with tracking
            detections = detector.detect_with_tracking(frame)
            
            # Process operators through multi-operator monitor
            active_operators = operator_monitor.process_frame(frame, detections, gray)
            
            # ============= JOIN/LEAVE NOTIFICATIONS =============
            current_operators = set()
            operator_states = {}
            
            # Track current operators and their states
            for op in active_operators:
                op_id = f"op_{op['slot']}"
                current_operators.add(op_id)
                operator_states[op_id] = {
                    'slot': op['slot'],
                    'state': op['state'],
                    'idle_duration': op['idle_duration']
                }
            
            # Check for operators who JOINED
            for op_id in current_operators:
                if op_id not in previous_operators:
                    # New operator joined - send notification with current state
                    slot = operator_states[op_id]['slot']
                    state = operator_states[op_id]['state']
                    add_notification(
                        f"👤 Operator #{slot} joined - {state}", 
                        "success", 
                        operator_id=op_id, 
                        state=f"JOINED_{state}"
                    )
            
            # Check for operators who LEFT
            for op_id in previous_operators:
                if op_id not in current_operators:
                    # Operator left - reset their notification tracking
                    slot = op_id.split('_')[1]
                    add_notification(
                        f"🚪 Operator #{slot} left", 
                        "warning",
                        operator_id=op_id,
                        state="LEFT"
                    )
                    # Reset notification tracking for this operator
                    reset_operator_notifications(op_id)
            
            previous_operators = current_operators.copy()
            # ====================================================
            
            # Update motion for each operator
            for op in active_operators:
                slot = op['slot'] - 1
                x1, y1, x2, y2 = op['box']
                op_id = f"op_{op['slot']}"
                
                if prev_gray is not None:
                    h, w = gray.shape
                    y1 = max(0, min(y1, h-1))
                    y2 = max(y1+1, min(y2, h))
                    x1 = max(0, min(x1, w-1))
                    x2 = max(x1+1, min(x2, w))
                    
                    if y2 > y1 and x2 > x1:
                        person_roi = gray[y1:y2, x1:x2]
                        prev_roi = prev_gray[y1:y2, x1:x2]
                        
                        if person_roi.size > 0 and prev_roi.size > 0:
                            diff = cv2.absdiff(prev_roi, person_roi)
                            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                            
                            motion_area = np.count_nonzero(thresh)
                            motion_intensity = np.sum(diff)
                            
                            # Check if motion is significant
                            if motion_area > MIN_MOTION_AREA and motion_intensity > MOTION_THRESHOLD:
                                state_changed = operator_monitor.update_operator_state(slot, True, current_time)
                                if state_changed:
                                    # Operator became ACTIVE - send notification
                                    add_notification(
                                        f"⚡ Operator #{op['slot']} is now ACTIVE", 
                                        "success",
                                        operator_id=op_id,
                                        state="ACTIVE"
                                    )
                            else:
                                old_state = operator_monitor.operator_data[slot]['state'] if operator_monitor.operator_data[slot] else None
                                state_changed = operator_monitor.update_operator_state(slot, False, current_time)
                                
                                if state_changed and operator_monitor.operator_data[slot]['state'] == 'IDLE':
                                    # Operator became IDLE - send notification
                                    add_notification(
                                        f"😴 Operator #{op['slot']} is now IDLE", 
                                        "warning",
                                        operator_id=op_id,
                                        state="IDLE"
                                    )
            
            prev_gray = gray.copy()
            
            # Draw bounding boxes on frame
            frame_with_boxes = draw_bounding_boxes(frame.copy(), active_operators)
            
            # Get summary
            summary = operator_monitor.get_summary()
            
            # Determine system state
            if summary['occupied_slots'] == 0:
                if no_operator_start_time is None:
                    no_operator_start_time = current_time
                    add_notification("👥 All operators left the frame", "warning")
                
                no_operator_duration = current_time - no_operator_start_time
                
                if no_operator_duration > ABSENT_TIME:
                    state = "ABSENT"
                    
                    if not no_operator_warning_shown:
                        add_notification("🚨 CRITICAL: No operators detected!", "critical")
                        no_operator_warning_shown = True
                else:
                    state = "ACTIVE"
            else:
                if no_operator_start_time is not None:
                    add_notification("👥 Operators have returned to frame", "success")
                    no_operator_warning_shown = False
                
                no_operator_start_time = None
                
                if summary['idle_count'] > 0 and summary['active_count'] == 0:
                    state = "IDLE"
                    if prev_state != "IDLE":
                        add_notification("😴 All operators are now IDLE", "warning")
                elif summary['active_count'] > 0:
                    state = "ACTIVE"
                    if prev_state == "IDLE":
                        add_notification("⚡ Operators are now ACTIVE", "success")
                else:
                    state = "UNKNOWN"
                last_seen = current_time
            
            # Update state if changed
            with state_lock:
                if state != prev_state:
                    logger.log(state)
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"📝 [{timestamp}] System State: {prev_state} -> {state}")
                    
                    prev_state = state
                    current_state = state
                    state_start_time = current_time
                    
                    state_history.append({
                        'state': state,
                        'time': timestamp
                    })
                    if len(state_history) > 100:
                        state_history = state_history[-100:]
            
            # Update dashboard with frame
            with frame_lock:
                latest_frame = frame_with_boxes
            
            # Show window
            cv2.imshow('Multi-Operator Monitor', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        add_notification("🛑 System stopped by user", "info")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        add_notification(f"❌ System error: {str(e)}", "critical")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
