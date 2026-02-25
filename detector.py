import cv2
from ultralytics import YOLO

class HumanDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.model.conf = 0.3
        
    def detect_with_tracking(self, frame):
        """Detect and track multiple persons (max 4)"""
        results = self.model.track(frame, persist=True, verbose=False)[0]
        
        detections = []
        if results.boxes and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            
            # Limit to first 4 detections
            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                if i >= 4:  # Max 4 operators
                    break
                    
                if int(results.boxes.cls[i]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, box[:4])
                    detections.append({
                        'id': int(track_id),
                        'box': (x1, y1, x2, y2)
                    })
        
        return detections