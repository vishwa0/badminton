import cv2
import numpy as np
import pickle
import torch
from src.core.face_recognizer import FaceRecognizer
from src.utils.drawing_utils import Drawer
from src.core.player_detector import PlayerDetector
from src.core.tracker import MultiObjectTracker
from src.utils.video_utils import VideoProcessor
import os

def run_test(video_path):
    """Runs a test of the face recognition and drawing functionality on a video."""

    # Create a directory to save debug face crops
    debug_dir = "debug_faces"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    extracted_dir = "extracted_faces"
    if not os.path.exists(extracted_dir):
        os.makedirs(extracted_dir)

    # Initialize components
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    player_detector = PlayerDetector(device=device)
    face_recognizer = FaceRecognizer(encodings_path="arcface_encodings.pkl", device=device)
    tracker = MultiObjectTracker()
    drawer = Drawer()
    video_processor = VideoProcessor(video_path)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_output.avi', fourcc, 20.0, (1280, 720))

    frame_count = 0
    booker_found = False
    authorized_coaches = ["Krithika"]
    frame_skip = 2 # Skip every 2nd frame
    for frame in video_processor:
        if frame is None or frame_count > 400:
            break
        
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        frame_count += 1

        coach_present_in_frame = False

        # Player detection
        players = player_detector.detect_players(frame, None)

        # Track players
        tracked_players = tracker.update(players)

        # Recognize faces for new tracks
        for player in tracked_players:
            if 'name' not in player:
                x1, y1, x2, y2 = player['bbox']

                # Crop to the upper part of the bounding box for face recognition
                # Use the full player bounding box for better face detection
                player_img = frame[y1:y2, x1:x2]

                # Recognize the face and store the name
                # Save the cropped face for debugging
                track_id = player.get('id', frame_count)
                debug_image_path = os.path.join(debug_dir, f"frame_{frame_count}_track_{track_id}.jpg")
                if player_img.size > 0:
                    cv2.imwrite(debug_image_path, player_img)

                result = face_recognizer.recognize_face(player_img)
                player['name'] = result['name']
                player['confidence'] = result['confidence']
                
                if player['name'] != "Unknown":
                    print(f"Recognized {player['name']} with confidence {player['confidence']:.2f}")
                    if not booker_found:
                        booker_found = True
                    if player['name'] in authorized_coaches:
                        coach_present_in_frame = True


        # Draw results
        output_frame = drawer.draw_all(frame.copy(), None, tracked_players, None)

        if booker_found:
            cv2.putText(output_frame, "Court booked person is present", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if coach_present_in_frame:
            cv2.putText(output_frame, "Authorized coach is found", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write the frame
        out.write(cv2.resize(output_frame, (1280, 720)))

        # Display the live tracker
        cv2.imshow('Live Tracker', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print("Test complete. Check 'test_output.avi' for results.")

if __name__ == "__main__":
    video_path = r"WhatsApp Video 2025-08-09 at 23.06.13.mp4"

    run_test(video_path)
