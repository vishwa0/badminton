import streamlit as st
import cv2
import tempfile
import time
import threading
from src.core.player_detector import PlayerDetector
from src.core.court_detector import CourtDetector
from src.core.coaching_detector import CoachingDetector
from src.core.shuttlecock_detector import ShuttlecockDetector
from src.core.tracker import MultiObjectTracker
from src.core.analyzer import MatchAnalyzer
from src.utils.video_utils import VideoProcessor
from src.utils.drawing_utils import Drawer
from src.core.face_recognizer import FaceRecognizer
from src.utils.file_utils import clear_directory

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    load_css("assets/style.css")

    st.markdown("<h1>Badminton Court Monitoring</h1>", unsafe_allow_html=True)

    # Initialize Face Recognizer
    if 'face_recognizer' not in st.session_state:
        st.session_state.face_recognizer = FaceRecognizer()

    # The player registration UI has been removed as per the new requirements.

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Initialize all components
        court_detector = CourtDetector()
        player_detector = PlayerDetector()
        shuttlecock_detector = ShuttlecockDetector()
        coaching_detector = CoachingDetector()
        tracker = MultiObjectTracker(iou_threshold=0.025)
        analyzer = MatchAnalyzer()
        drawer = Drawer()
        face_recognizer = st.session_state.face_recognizer
        
        video_processor = VideoProcessor(video_path)
        
        # Run coaching detection in a separate thread
        coaching_thread = threading.Thread(target=coaching_detector.run_analysis, args=(video_path,))
        coaching_thread.start()

        stframe = st.empty()
        player_count_placeholder = st.empty()
        
        # Create columns for the cards
        col1, col2, col3 = st.columns(3)
        with col1:
            booker_presence_placeholder = st.empty()
        with col2:
            authorized_coach_placeholder = st.empty()
        with col3:
            coaching_status_placeholder = st.empty()

        # Flags for sticky status
        booker_present_detected = False
        authorized_coach_name = None

        for frame in video_processor:

            # Court detection
            court_mask, court_corners = court_detector.detect_court(frame)
            
            # Player detection
            players = player_detector.detect_players(frame, court_corners)
            
            # Shuttlecock detection
            shuttlecock_pos = shuttlecock_detector.detect(frame)
            
            # Multi-object tracking
            tracked_players = tracker.update(players)

            # Face Recognition
            for player in tracked_players:
                if 'name' not in player:
                    x1, y1, x2, y2 = player['bbox']
                    player_img = frame[y1:y2, x1:x2]
                    result = face_recognizer.recognize_face(player_img)
                    player['name'] = result['name']
                    player['confidence'] = result['confidence']
            
            # Analysis
            analyzer.update(tracked_players, shuttlecock_pos, court_corners)
            players_in_court = analyzer.get_players_in_court()
            player_counts = analyzer.get_player_counts()

            # Visualization
            # Visualization
            output_frame = drawer.draw_all(frame, court_corners, players_in_court, shuttlecock_pos, player_counts, tracked_players)
            
            stframe.image(output_frame, channels="BGR", use_container_width=True)
            total_players = player_counts['side_a'] + player_counts['side_b']
            player_count_placeholder.markdown(f"""
                <div class='player-count'>
                    Total Players: {total_players}
                </div>
            """, unsafe_allow_html=True)

            # --- Sticky Status Logic ---
            if not booker_present_detected:
                if any(p['name'] != 'Unknown' for p in tracked_players):
                    booker_present_detected = True

            if not authorized_coach_name:
                coach = next((p['name'] for p in tracked_players if p['name'] == 'Krithika'), None)
                if coach:
                    authorized_coach_name = coach
            
            coaching_detected = coaching_detector.get_result() == "yes"
            print("----------------------------------",coaching_detected,"-----------------------------------------------------")

            # --- Update UI Cards ---

            # Booker Presence Card
            if booker_present_detected:
                booker_presence_placeholder.markdown("""
                    <div class='status-card status-present'>
                        <strong>Booker Presence</strong><br>Detected
                    </div>
                """, unsafe_allow_html=True)
            else:
                booker_presence_placeholder.markdown("""
                    <div class='status-card status-absent'>
                        <strong>Booker Presence</strong><br>Not Detected
                    </div>
                """, unsafe_allow_html=True)

            # Authorized Coach Card
            if authorized_coach_name:
                authorized_coach_placeholder.markdown(f"""
                    <div class='status-card status-present'>
                        <strong>Authorized Coach</strong><br>{authorized_coach_name}
                    </div>
                """, unsafe_allow_html=True)
            else:
                authorized_coach_placeholder.markdown("""
                    <div class='status-card status-absent'>
                        <strong>Authorized Coach</strong><br>Not Detected
                    </div>
                """, unsafe_allow_html=True)

            # Coaching Detection Card
            if coaching_detected:
                if authorized_coach_name:
                    coaching_status_placeholder.markdown(f"""
                        <div class='status-card status-present'>
                            <strong>Coaching</strong><br>Authorized
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    coaching_status_placeholder.markdown("""
                        <div class='status-card status-warning'>
                            <strong>Coaching</strong><br>Unauthorized
                        </div>
                    """, unsafe_allow_html=True)
            else:
                coaching_status_placeholder.markdown("""
                    <div class='status-card status-absent'>
                        <strong>Coaching</strong><br>Not Detected
                    </div>
                """, unsafe_allow_html=True)
        
        coaching_thread.join()

        # Clear the debug and extracted faces directories
        clear_directory("debug_faces")
        clear_directory("extracted_faces")

        st.success("Video processing complete and temporary files cleared.")


if __name__ == '__main__':
    main()