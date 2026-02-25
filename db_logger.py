import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_path = csv_path
        
        # Create file with headers if not exists
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'state'])
    
    def log(self, state):
        """Log state change"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), state])