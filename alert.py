import logging
from datetime import datetime

class Alert:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.last_alert_time = {}
        self.alert_cooldown = 60  # Don't repeat same alert within 60 seconds
    
    def send(self, message, alert_type="INFO"):
        """Send alert with different severity levels"""
        current_time = datetime.now()
        
        # Check cooldown for repeated alerts
        if message in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[message]).total_seconds()
            if time_diff < self.alert_cooldown:
                return  # Skip alert if within cooldown
        
        # Log with appropriate level
        if alert_type == "CRITICAL" or "🚨" in message:
            logging.critical(f"🔴 {message}")
        elif alert_type == "WARNING" or "⚠️" in message:
            logging.warning(f"🟡 {message}")
        else:
            logging.info(f"✅ {message}")
        
        # Update last alert time
        self.last_alert_time[message] = current_time
        
        # Here you could also add:
        # - Email alerts
        # - SMS alerts
        # - Slack notifications
        # - Desktop notifications
    
    def send_critical(self, message):
        """Send critical alert"""
        self.send(message, "CRITICAL")
    
    def send_warning(self, message):
        """Send warning alert"""
        self.send(message, "WARNING")
    
    def clear_cooldown(self, message):
        """Clear cooldown for a specific alert"""
        if message in self.last_alert_time:
            del self.last_alert_time[message]