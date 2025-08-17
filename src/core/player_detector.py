import torch
import torchvision
from ultralytics import YOLO
import cv2
import numpy as np

class PlayerDetector:
    def __init__(self, model_path=None, device=None):
        """Initialize YOLO model for player detection."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path or 'yolov8n.pt')
        self.model.to(self.device)

    def is_inside_court(self, bbox, court_corners):
        """Check if the center of a bounding box is inside the court polygon."""
        if court_corners is None:
            return False
        
        x1, y1, x2, y2 = bbox
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Ensure court_corners is a NumPy array of shape (4, 2) and correct type
        court_polygon = np.array(court_corners, dtype=np.int32).reshape(4, 2)
        
        return cv2.pointPolygonTest(court_polygon, center, False) >= 0

    def detect_players(self, frame, court_corners=None):
        """Detect players, optionally filtering by court boundaries."""
        results = self.model(frame, device=self.device)
        players = []

        if not results:
            return []
        
        result = results[0]
        
        boxes_for_nms = []
        confidences_for_nms = []

        if result.boxes is None:
            return []
            
        frame_height = frame.shape[0]
        for box in result.boxes:
            if box.cls == 0:  # Person class
                if box.conf[0] > 0.7: # Increased confidence threshold
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    bbox_height = y2 - y1

                    # Filter based on size to remove distant people
                    if bbox_height > frame_height * 0.1:
                        # If court_corners are provided, check if the player is inside
                        if court_corners is None or self.is_inside_court(bbox, court_corners):
                            boxes_for_nms.append(torch.tensor(bbox, dtype=torch.float32))
                            confidences_for_nms.append(box.conf[0])
        
        if not boxes_for_nms:
            return []

        # Apply Non-Maximum Suppression
        boxes_tensor = torch.stack(boxes_for_nms).to(self.device)
        confidences_tensor = torch.stack(confidences_for_nms).to(self.device)
        
        indices = torchvision.ops.nms(boxes_tensor, confidences_tensor, iou_threshold=0.3)
        
        final_boxes = boxes_tensor[indices]
        final_confidences = confidences_tensor[indices]

        for i in range(len(final_boxes)):
            x1, y1, x2, y2 = final_boxes[i].cpu().numpy()
            confidence = final_confidences[i].cpu().numpy()
            players.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(confidence)
            })
        
        return players