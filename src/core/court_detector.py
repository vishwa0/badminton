import cv2
import numpy as np
from collections import deque

class CourtDetector:
    def __init__(self, history_length=10):
        self.court_corners = None
        self.homography = None
        self.corner_history = deque(maxlen=history_length)

    def detect_court(self, frame):
        """Detect court using a refined hybrid line and contour approach"""
        corners = self.detect_with_lines(frame)
        
        if corners is None:
            corners = self.detect_with_contours(frame)

        if corners is not None:
            self.corner_history.append(corners)
            avg_corners = np.mean(self.corner_history, axis=0).astype(np.float32)
            self.court_corners = avg_corners
            self.homography = self.compute_homography(self.court_corners)
        
        return frame, self.court_corners

    def detect_with_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 100) # Adjusted thresholds
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=30)

        if lines is None:
            return None

        h_lines, v_lines = self.group_lines(lines)
        if not h_lines or not v_lines:
            return None

        intersections = self.find_intersections(h_lines, v_lines)
        if len(intersections) < 4:
            return None

        return self.find_corners_from_points(intersections)

    def detect_with_contours(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Filter contours by area
        min_court_area = frame.shape[0] * frame.shape[1] * 0.2 # At least 20% of frame
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_court_area]
        
        if not valid_contours:
            return None

        largest_contour = max(valid_contours, key=cv2.contourArea)
        return self.find_corners_from_points(largest_contour.reshape(-1, 2))

    def group_lines(self, lines):
        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 45 or angle > 135: v_lines.append(line[0])
            else: h_lines.append(line[0])
        return h_lines, v_lines

    def find_intersections(self, h_lines, v_lines):
        intersections = []
        for h_line in h_lines:
            for v_line in v_lines:
                pt = self.line_intersection(h_line, v_line)
                if pt: intersections.append(pt)
        return intersections

    def line_intersection(self, line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0: return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return [int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))]
        return None

    def find_corners_from_points(self, points):
        if len(points) < 4: return None
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
        
        if len(approx) == 4:
            return self.order_points(approx.reshape(4, 2))
        else:
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            return self.order_points(box.astype(int))

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def compute_homography(self, corners):
        if corners is None: return None
        real_court = np.array([[0, 0], [610, 0], [610, 1340], [0, 1340]], dtype=np.float32)
        h, _ = cv2.findHomography(corners, real_court)
        return h