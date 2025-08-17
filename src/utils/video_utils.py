import cv2

class VideoProcessor:
    def __init__(self, video_path):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(video_path)

    def __iter__(self):
        return self

    def __next__(self):
        """Get next frame from video"""
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def get_fps(self):
        """Return video's frames per second"""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_dims(self):
        """Return video dimensions (width, height)"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def release(self):
        """Release video capture resources"""
        self.cap.release()