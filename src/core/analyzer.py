import cv2
import numpy as np

class MatchAnalyzer:
    def __init__(self):
        self.player_counts = {'side_a': 0, 'side_b': 0}
        self.players_in_court = []

    def update(self, tracked_players, shuttlecock_pos, court_corners):
        """Analyze current frame and update player counts"""
        self.players_in_court = tracked_players
        
        if court_corners is None:
            self.player_counts = {'side_a': 0, 'side_b': 0}
            return

        # Calculate court center line for side assignment
        y_top = court_corners[0][1]
        y_bottom = court_corners[2][1]
        center_y = (y_top + y_bottom) / 2

        # Assign players to sides based on position relative to center line
        side_a = [p for p in self.players_in_court 
                 if (p['bbox'][1] + p['bbox'][3]) / 2 < center_y]
        side_b = [p for p in self.players_in_court 
                 if (p['bbox'][1] + p['bbox'][3]) / 2 >= center_y]

        self.player_counts = {'side_a': len(side_a), 'side_b': len(side_b)}

    def get_player_counts(self):
        """Return current player counts for each side"""
        return self.player_counts

    def get_players_in_court(self):
        """Return the list of players inside the court"""
        return self.players_in_court