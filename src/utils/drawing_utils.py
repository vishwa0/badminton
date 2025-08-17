import cv2

class Drawer:
    def draw_court(self, frame, court_corners):
        """Draw court corners"""
        if court_corners is not None:
            for corner in court_corners:
                cv2.circle(frame, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
        return frame

    def draw_shuttlecock(self, frame, shuttlecock):
        """Draw shuttlecock position"""
        if shuttlecock is not None:
            cv2.circle(frame, shuttlecock, 7, (0, 0, 255), -1)
        return frame

    def draw_players(self, frame, players):
        """Draw player bounding boxes and IDs/Names"""
        for player in players:
            bbox = player['bbox']
            # The drawing of player names and IDs has been removed as per requirements.
            # We will only draw the bounding box now.
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        return frame

    def draw_all(self, frame, court_corners, players, shuttlecock, player_counts=None, recognized_players=None):
        """Draw all detections and analysis results on frame"""
        frame = self.draw_court(frame, court_corners)
        frame = self.draw_players(frame, players)
        frame = self.draw_shuttlecock(frame, shuttlecock)
        return frame