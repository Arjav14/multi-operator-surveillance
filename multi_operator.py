import cv2
import time
import numpy as np
from datetime import datetime

class MultiOperatorMonitor:
    def __init__(self, max_operators=4, idle_threshold=10, absence_threshold=5):
        self.max_operators = max_operators
        self.idle_threshold = idle_threshold
        self.absence_threshold = absence_threshold
        
        # Fixed-size arrays for 4 operators
        self.operators = [None] * max_operators  # None means slot available
        self.operator_data = [None] * max_operators  # Current data for active operators
        self.operator_history = [[] for _ in range(max_operators)]  # History per slot
        
        # Tracking
        self.next_slot = 0
        self.total_operators_seen = 0
        
        # Motion settings
        self.motion_threshold = 15000
        self.min_motion_area = 500
        
        print(f"✅ Multi-Operator Monitor initialized")
        print(f"   Max operators: {max_operators}")
        print(f"   Idle threshold: {idle_threshold}s")
        print(f"   Absence threshold: {absence_threshold}s")
        print("-" * 40)
    
    def find_empty_slot(self):
        """Find first empty slot for new operator"""
        for i in range(self.max_operators):
            if self.operators[i] is None:
                return i
        return -1  # No empty slots
    
    def find_operator_by_id(self, track_id):
        """Find operator slot by track ID"""
        for i in range(self.max_operators):
            if self.operators[i] == track_id:
                return i
        return -1
    
    def assign_new_operator(self, track_id, box, current_time):
        """Assign new operator to empty slot"""
        slot = self.find_empty_slot()
        if slot == -1:
            # No empty slots, replace oldest idle/absent operator
            slot = self.find_oldest_idle()
            if slot == -1:
                slot = self.find_oldest_active()
        
        if slot != -1:
            # Remove old operator from this slot
            if self.operators[slot] is not None:
                print(f"🔄 Slot {slot+1}: Replaced operator {self.operators[slot]}")
            
            # Assign new operator
            self.operators[slot] = track_id
            self.operator_data[slot] = {
                'track_id': track_id,
                'first_seen': current_time,
                'last_seen': current_time,
                'last_motion': current_time,
                'position': box,
                'state': 'ACTIVE',
                'state_start': current_time,
                'total_active': 0,
                'total_idle': 0,
                'prev_gray': None
            }
            self.total_operators_seen += 1
            print(f"✅ Slot {slot+1}: New operator {track_id}")
            return slot
        
        return -1
    
    def find_oldest_idle(self):
        """Find slot with longest idle operator to replace"""
        oldest_time = float('inf')
        oldest_slot = -1
        
        for i in range(self.max_operators):
            if self.operator_data[i] is not None:
                if self.operator_data[i]['state'] == 'IDLE':
                    idle_duration = time.time() - self.operator_data[i]['last_motion']
                    if idle_duration < oldest_time:
                        oldest_time = idle_duration
                        oldest_slot = i
        
        return oldest_slot
    
    def find_oldest_active(self):
        """Find slot with oldest active operator as last resort"""
        oldest_time = float('inf')
        oldest_slot = 0
        
        for i in range(self.max_operators):
            if self.operator_data[i] is not None:
                if self.operator_data[i]['first_seen'] < oldest_time:
                    oldest_time = self.operator_data[i]['first_seen']
                    oldest_slot = i
        
        return oldest_slot
    
    def update_operator_state(self, slot, motion_detected, current_time):
        """Update state for operator in given slot"""
        if slot < 0 or slot >= self.max_operators:
            return False
        
        op_data = self.operator_data[slot]
        if op_data is None:
            return False
        
        old_state = op_data['state']
        
        # Update motion time if motion detected
        if motion_detected:
            op_data['last_motion'] = current_time
            # If motion detected, operator is definitely active
            if old_state != 'ACTIVE':
                op_data['state'] = 'ACTIVE'
                op_data['state_start'] = current_time
                print(f"👤 Slot {slot+1}: Motion detected -> ACTIVE")
                return True
        else:
            # No motion detected, check if should be idle
            time_since_motion = current_time - op_data['last_motion']
            
            # Only transition to IDLE if enough time has passed
            if time_since_motion > self.idle_threshold and old_state != 'IDLE':
                op_data['state'] = 'IDLE'
                op_data['state_start'] = current_time
                print(f"👤 Slot {slot+1}: No motion for {time_since_motion:.1f}s -> IDLE")
                return True
        
        return False
    
    def process_frame(self, frame, detections, grayscale=None):
        """Process frame with multiple operator detections"""
        current_time = time.time()
        active_operators = []
        
        # Track which slots are active this frame
        active_slots = []
        
        # Process each detection
        for detection in detections:
            track_id = detection['id']
            box = detection['box']
            
            # Find if operator already exists
            slot = self.find_operator_by_id(track_id)
            
            if slot == -1:
                # New operator - assign to slot
                slot = self.assign_new_operator(track_id, box, current_time)
            
            if slot != -1:
                # Update operator data
                self.operator_data[slot]['last_seen'] = current_time
                self.operator_data[slot]['position'] = box
                active_slots.append(slot)
                
                # Calculate idle duration for this operator
                time_since_motion = current_time - self.operator_data[slot]['last_motion']
                state_duration = current_time - self.operator_data[slot]['state_start']
                
                active_operators.append({
                    'slot': slot + 1,  # 1-based for display
                    'id': track_id,
                    'box': box,
                    'state': self.operator_data[slot]['state'],
                    'idle_duration': int(time_since_motion),
                    'state_duration': int(state_duration)
                })
        
        # Check for operators who left (absent)
        for slot in range(self.max_operators):
            if self.operator_data[slot] is not None and slot not in active_slots:
                op_data = self.operator_data[slot]
                time_absent = current_time - op_data['last_seen']
                
                if time_absent > self.absence_threshold:
                    # Operator is absent - update stats
                    duration = current_time - op_data['state_start']
                    
                    if op_data['state'] == 'ACTIVE':
                        op_data['total_active'] += duration
                    elif op_data['state'] == 'IDLE':
                        op_data['total_idle'] += duration
                    
                    print(f"🚪 Slot {slot+1} (ID:{op_data['track_id']}): ABSENT")
                    
                    # Keep in system but mark as None for slot availability
                    self.operators[slot] = None
                    self.operator_data[slot] = None
        
        return active_operators
    
    def get_summary(self):
        """Get current status summary"""
        current_time = time.time()
        
        active_count = 0
        idle_count = 0
        operator_details = []
        
        for slot in range(self.max_operators):
            if self.operator_data[slot] is not None:
                op_data = self.operator_data[slot]
                
                if op_data['state'] == 'ACTIVE':
                    active_count += 1
                elif op_data['state'] == 'IDLE':
                    idle_count += 1
                
                # Calculate durations
                state_duration = int(current_time - op_data['state_start'])
                time_since_motion = int(current_time - op_data['last_motion'])
                
                operator_details.append({
                    'slot': slot + 1,
                    'id': op_data['track_id'],
                    'state': op_data['state'],
                    'state_duration': state_duration,
                    'idle_duration': time_since_motion,
                    'position': op_data['position']
                })
        
        return {
            'total_slots': self.max_operators,
            'occupied_slots': len(operator_details),
            'active_count': active_count,
            'idle_count': idle_count,
            'operators': operator_details,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
    
    def get_statistics(self):
        """Get detailed statistics per operator"""
        stats = []
        
        for slot in range(self.max_operators):
            if self.operator_data[slot] is not None:
                op_data = self.operator_data[slot]
                total_time = op_data['total_active'] + op_data['total_idle']
                
                stats.append({
                    'slot': slot + 1,
                    'id': op_data['track_id'],
                    'first_seen': datetime.fromtimestamp(op_data['first_seen']).strftime("%H:%M:%S"),
                    'current_state': op_data['state'],
                    'active_time': int(op_data['total_active']),
                    'idle_time': int(op_data['total_idle']),
                    'active_percent': round((op_data['total_active'] / total_time * 100), 1) if total_time > 0 else 0
                })
        
        return stats
    
    def get_history(self, slot=None):
        """Get history for specific slot or all slots"""
        if slot is not None and 0 <= slot < self.max_operators:
            return self.operator_history[slot]
        
        # Return all history
        all_history = []
        for s in range(self.max_operators):
            for event in self.operator_history[s]:
                event_copy = event.copy()
                event_copy['slot'] = s + 1
                all_history.append(event_copy)
        
        # Sort by time (most recent first)
        all_history.sort(key=lambda x: x['time'], reverse=True)
        return all_history[:50]  # Last 50 events