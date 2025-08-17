import cv2
import os
import random
import google.generativeai as genai
import sys
import math
import time
from dotenv import load_dotenv

class CoachingDetector:
    def __init__(self):
        self.result = None

    def get_result(self):
        return self.result

    def _get_video_duration(self, video_path):
        """Gets the duration of a video in seconds."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration

    def _extract_clip(self, video_path, start_time, duration, output_path):
        """Extracts a clip from a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        
        end_time = start_time + duration
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > end_time:
                break
            out.write(frame)
            
        cap.release()
        out.release()
        return True

    def _analyze_video_with_gemini(self, video_path):
        """Analyzes video with Gemini by sending raw bytes."""
        load_dotenv(dotenv_path="src/core/.env")
        if "GEMINI_API_KEY" not in os.environ:
            raise ValueError("GEMINI_API_KEY not found, please set it in src/core/.env file")
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        print(f"Reading video file: {video_path}")
        with open(video_path, 'rb') as f:
            video_bytes = f.read()

        video_part = {
            "mime_type": "video/mp4",
            "data": video_bytes
        }

        prompt = """
You are a highly experienced badminton coach acting as a court monitor. Your task is to determine if any coaching activities are taking place in the provided video.

INSTRUCTIONS:
1.  Analyze the video for patterns of one-sided drills where one person is consistently feeding the shuttlecock to the other.
2.  Look for repetitive exercises targeting specific shots (e.g., smashes, drop shots) rather than a regular game.
3.  A regular match involves rallies between two players. Coaching involves one person (the coach) setting up drills for the other (the player).
4.  Ignore warm-up activities and focus on structured, repetitive drills that are characteristic of a coaching session.
5.  Your response must be a single word: "yes" if coaching is detected, and "no" if it is not.
6.  Do not include any additional text or explanations in your response.

Ensure that your final response doest have any additional text than "yes" or "no".The output should be in exactly one word. ONE WORD

EXAMPLE OUTPUT if coaching activities present in the video:
yes

Example OUTPUT if no coaching activities detected:
no
"""

        model = genai.GenerativeModel(model_name="gemini-2.0-flash-lite")
        print("Making LLM inference request with video bytes...")
        response = model.generate_content([prompt, video_part], request_options={"timeout": 600})
        
        return response.text.strip().lower()

    def run_analysis(self, video_path):
        """Main function to process the video."""
        duration = self._get_video_duration(video_path)
        print(f"Video duration: {duration:.2f} seconds")

        temp_files = []
        analysis_result = "--"  # Default to no coaching

        try:
            if duration <= 120:
                print("Video is 2 minutes or less. Analyzing the full video.")
                analysis_result = self._analyze_video_with_gemini(video_path)
            elif duration <= 600:
                print("Video is between 2 and 10 minutes. Analyzing a random 2-minute clip.")
                start_time = random.uniform(0, duration - 120)
                output_path = f"temp_clip_{int(time.time())}.mp4"
                temp_files.append(output_path)
                if self._extract_clip(video_path, start_time, 120, output_path):
                    analysis_result = self._analyze_video_with_gemini(output_path)
            else:
                print("Video is longer than 10 minutes. Splitting into 5-minute chunks and analyzing a random 2-minute clip from each.")
                num_chunks = math.ceil(duration / 300)
                for i in range(num_chunks):
                    chunk_start_time = i * 300
                    chunk_duration = min(300, duration - chunk_start_time)
                    if chunk_duration > 120:
                        start_time = random.uniform(chunk_start_time, chunk_start_time + chunk_duration - 120)
                        output_path = f"temp_clip_{int(time.time())}_{i}.mp4"
                        temp_files.append(output_path)
                        if self._extract_clip(video_path, start_time, 120, output_path):
                            print(f"\nAnalyzing clip from chunk {i+1}...")
                            result_chunk = self._analyze_video_with_gemini(output_path)
                            print(f"Coaching detected in chunk {i+1}: {result_chunk}")
                            if result_chunk == "yes":
                                analysis_result = "yes"
                                print("Coaching detected. Stopping analysis.")
                                break  # Stop if coaching is detected in any chunk
        finally:
            # Clean up temporary files
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"Removed temporary file: {f}")
            
            self.result = analysis_result
            print(f"Coaching detected: {self.result}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/core/coaching_detector.py <path_to_video>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
        
    detector = CoachingDetector()
    detector.run_analysis(video_path)