import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace
import argparse

def generate_encodings(image_dir, output_path, model_name='ArcFace'):
    """
    Generates ArcFace embeddings for faces in the specified image directory
    and saves them to a pickle file.

    Args:
        image_dir (str): Path to the directory containing subdirectories of known faces.
                         Each subdirectory name will be used as the person's name.
        output_path (str): Path to save the generated encodings.
        model_name (str): The name of the DeepFace model to use for embeddings (e.g., 'ArcFace').
    """
    known_names = []
    known_embeddings = []

    if not os.path.exists(image_dir):
        print(f"Error: Image directory '{image_dir}' not found.")
        return

    for person_name in os.listdir(image_dir):
        person_dir = os.path.join(image_dir, person_name)
        if os.path.isdir(person_dir):
            print(f"Processing images for: {person_name}")
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # DeepFace.represent can take a path directly
                        embeddings = DeepFace.represent(
                            img_path=image_path,
                            model_name=model_name,
                            enforce_detection=True  # Enforce face detection for known faces
                        )
                        if embeddings:
                            # Assuming one face per image for known faces
                            known_names.append(person_name)
                            known_embeddings.append(embeddings[0]['embedding'])
                            print(f"  - Added embedding for {image_name}")
                        else:
                            print(f"  - No face detected in {image_name}")
                    except Exception as e:
                        print(f"  - Error processing {image_name}: {e}")

    if known_names and known_embeddings:
        data = {"names": known_names, "encodings": known_embeddings}
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Successfully generated and saved {len(known_names)} encodings to {output_path}")
    else:
        print("No encodings were generated. Make sure faces are detectable in your images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate face encodings from a directory of images.')
    parser.add_argument('--imagedir', type=str, default='known_faces', help='Path to the directory of known faces.')
    parser.add_argument('--output', type=str, default='arcface_encodings.pkl', help='Path to save the encodings file.')
    args = parser.parse_args()
    generate_encodings(args.imagedir, args.output)