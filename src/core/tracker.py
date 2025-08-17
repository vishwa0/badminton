import numpy as np
from collections import deque

def iou(boxA, boxB):
    """Calculate Intersection over Union of two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)

class MultiObjectTracker:
    def __init__(self, iou_threshold=0.3, max_disappeared=20):
        self.next_id = 0
        self.tracks = {}
        self.disappeared = {}
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared

    def register(self, detection):
        """Register a new object"""
        track_id = self.next_id
        # Each track will store the full detection dictionary, which can include name, etc.
        self.tracks[track_id] = detection
        self.disappeared[track_id] = 0
        self.next_id += 1

    def deregister(self, track_id):
        """Deregister an object"""
        del self.tracks[track_id]
        del self.disappeared[track_id]

    def update(self, detections):
        """Update tracking with new detections"""
        if not detections:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister(track_id)
            return self.get_tracked_objects()

        if not self.tracks:
            for det in detections:
                self.register(det)
            return self.get_tracked_objects()

        track_ids = list(self.tracks.keys())
        track_bboxes = [t['bbox'] for t in self.tracks.values()]
        
        det_bboxes = [d['bbox'] for d in detections]
        
        iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)))
        for i, track_box in enumerate(track_bboxes):
            for j, det_box in enumerate(det_bboxes):
                iou_matrix[i, j] = iou(track_box, det_box)

        used_rows = set()
        used_cols = set()
        
        rows, cols = np.unravel_index(np.argsort(-iou_matrix, axis=None), iou_matrix.shape)
        
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            
            if iou_matrix[r, c] > self.iou_threshold:
                track_id = track_ids[r]
                
                # Preserve the name if it was already recognized
                existing_name = self.tracks[track_id].get('name')
                self.tracks[track_id] = detections[c]
                if existing_name:
                    self.tracks[track_id]['name'] = existing_name
                    
                self.disappeared[track_id] = 0
                used_rows.add(r)
                used_cols.add(c)

        unmatched_tracks = set(range(len(track_bboxes))) - used_rows
        unmatched_dets = set(range(len(det_bboxes))) - used_cols

        for r in unmatched_tracks:
            track_id = track_ids[r]
            self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                self.deregister(track_id)

        for c in unmatched_dets:
            self.register(detections[c])
            
        return self.get_tracked_objects()

    def get_tracked_objects(self):
        """Return a list of tracked objects with their IDs"""
        tracked_with_ids = []
        for track_id, track_data in self.tracks.items():
            data_with_id = track_data.copy()
            data_with_id['id'] = track_id
            tracked_with_ids.append(data_with_id)
        return tracked_with_ids