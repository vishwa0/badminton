import cv2
import numpy as np

class ShuttlecockDetector:
    def __init__(self):
        pass

    def detect(self, frame):
        """Detect shuttlecock position in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use HoughCircles to detect circular objects (shuttlecock)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=1, maxRadius=10
        )
        
        shuttlecock_pos = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return position of first detected circle
            shuttlecock_pos = (circles[0][0][0], circles[0][0][1])
        
        return shuttlecock_pos