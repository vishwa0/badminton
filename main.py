
import argparse
import cv2
from src.core.player_detector import PlayerDetector
from src.core.court_detector import CourtDetector
from src.core.shuttlecock_detector import ShuttlecockDetector
from src.core.tracker import MultiObjectTracker
from src.core.analyzer import MatchAnalyzer
from src.utils.video_utils import VideoProcessor
from src.utils.drawing_utils import Drawer

def main(video_path):
    # Initialize all components
    court_detector = CourtDetector()
    player_detector = PlayerDetector()
    shuttlecock_detector = ShuttlecockDetector()
    tracker = MultiObjectTracker(iou_threshold=0.025)
    analyzer = MatchAnalyzer()
    drawer = Drawer()
    
    video_processor = VideoProcessor(video_path)
    
    # Get video's native FPS and dimensions
    fps = video_processor.get_fps()
    width, height = video_processor.get_dims()
    wait_time = int(1000 / fps) if fps > 0 else 25

    # Create a named window that can be resized
    cv2.namedWindow('Badminton Court Monitoring', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Badminton Court Monitoring', width, height)

    for frame in video_processor:
        # Phase 1: Court detection and mapping
        court_mask, court_corners = court_detector.detect_court(frame)
        
        # Phase 2: Player detection
        players = player_detector.detect_players(frame, court_corners)
        
        # Phase 3: Shuttlecock detection
        shuttlecock_pos = shuttlecock_detector.detect(frame)
        
        # Phase 4: Multi-object tracking
        tracked_players = tracker.update(players)
        
        # Phase 5: Analysis and player counting
        analyzer.update(tracked_players, shuttlecock_pos, court_corners)
        
        # Get the players who are inside the court
        players_in_court = analyzer.get_players_in_court()

        # Visualization and output
        output_frame = drawer.draw_all(frame, court_corners, players_in_court, 
                                     shuttlecock_pos, analyzer.get_player_counts())
        
        cv2.imshow('Badminton Court Monitoring', output_frame)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    
    video_processor.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Badminton court monitoring system')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    args = parser.parse_args()
    main(args.video)