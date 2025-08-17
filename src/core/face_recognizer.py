import cv2
import numpy as np
import torch
from deepface import DeepFace
import pickle

import os

class FaceRecognizer:
    def __init__(self, model_name='ArcFace', distance_metric='cosine', encodings_path='arcface_encodings.pkl', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.known_faces = self.load_known_faces(encodings_path)
        self.face_count = 0
        self.sr = None
        # DeepFace will use the GPU if tensorflow-gpu is installed.
        # We can't explicitly move the DeepFace model to a device like with PyTorch.
        # However, we can ensure the environment is set up correctly.
        if self.device == 'cuda':
            try:
                # This will trigger TensorFlow to use the GPU
                import tensorflow as tf
                tf.config.list_physical_devices('GPU')
            except ImportError:
                print("TensorFlow not found. GPU acceleration for DeepFace might not be available.")
            except Exception as e:
                print(f"Error setting up GPU for DeepFace: {e}")
        
        self.initialize_sr_model()

    def load_known_faces(self, encodings_path):
        """Load known faces from a pickle file."""
        try:
            with open(encodings_path, 'rb') as f:
                data = pickle.load(f)
                
                names = data['names']
                embeddings = data['encodings']
                
                known_faces = {}
                for name, embedding in zip(names, embeddings):
                    if name not in known_faces:
                        known_faces[name] = []
                    known_faces[name].append(embedding)

                # Compute the average embedding for each person
                avg_known_faces = {}
                for name, embeddings_list in known_faces.items():
                    if embeddings_list:
                        avg_known_faces[name] = np.mean(embeddings_list, axis=0)
                
                return avg_known_faces
        except (FileNotFoundError, KeyError):
            return {}

    def initialize_sr_model(self):
        """Initializes the super-resolution model."""
        try:
            model_path = os.path.join(os.path.dirname(__file__), "EDSR_x4.pb")
            if not os.path.exists(model_path):
                print(f"Super-resolution model not found at {model_path}. Enhancement will be skipped.")
                return
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(model_path)
            self.sr.setModel("edsr", 4)
            print("Super-resolution model loaded successfully.")
        except Exception as e:
            print(f"Error initializing super-resolution model: {e}")
            self.sr = None

    def get_embedding(self, face_image):
        """Get ArcFace embedding for a face image."""
        try:
            # Ensure the input is a valid NumPy array with non-zero dimensions
            if not isinstance(face_image, np.ndarray) or face_image.size == 0 or any(dim == 0 for dim in face_image.shape):
                print(f"Invalid image with shape {face_image.shape} passed to get_embedding.")
                return None

            embedding = DeepFace.represent(
                img_path=face_image,
                model_name=self.model_name,
                enforce_detection=False,  # We've already detected the face
            )
            return embedding[0]['embedding']
        except Exception as e:
            # Log the error for debugging
            print(f"Error in get_embedding: {e}")
            return None

    def enhance_face_image(self, face_image):
        """Enhance the quality of a face image using a super-resolution model."""
        if face_image.size == 0:
            return face_image

        # Convert to uint8 for processing and returning
        if face_image.dtype != np.uint8:
            face_image = (face_image * 255).astype(np.uint8)

        if self.sr is None:
            return face_image  # Return uint8 image

        # Upscale the image
        upscaled_image = self.sr.upsample(face_image)

        # Apply a bilateral filter to reduce noise while preserving edges
        denoised_image = cv2.bilateralFilter(upscaled_image, d=9, sigmaColor=75, sigmaSpace=75)

        # Sharpen the image
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        sharpened_image = cv2.filter2D(denoised_image, -1, sharpening_kernel)

        return sharpened_image

    def recognize_face(self, player_img):
        """Recognizes a single face from a player image."""
        if player_img.size == 0:
            return {"name": "Unknown", "confidence": 0.0}

        try:
            # Detect faces within the player bounding box
            faces = DeepFace.extract_faces(
                img_path=player_img,
                enforce_detection=True,
                detector_backend='mtcnn'
            )

            if not faces:
                return {"name": "Unknown", "confidence": 0.0}

            best_match = "Unknown"
            best_similarity = 0.3  # Threshold for cosine similarity

            for face_data in faces:
                face_img = face_data['face']
                
                # Enhance the face image
                enhanced_face = self.enhance_face_image(face_img)

                # Save the extracted face
                face_filename = os.path.join("extracted_faces", f"face_{self.face_count}.jpg")
                if enhanced_face.size > 0:
                    img_to_save = enhanced_face
                    # DeepFace returns RGB, OpenCV uses BGR. Convert for saving.
                    if len(img_to_save.shape) > 2 and img_to_save.shape[2] == 3:
                        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(face_filename, img_to_save)
                    self.face_count += 1
                
                # Get the embedding for the detected face (use enhanced image)
                current_embedding = self.get_embedding(enhanced_face)
                if current_embedding is None:
                    continue

                for name, stored_embedding in self.known_faces.items():
                    current_embedding_np = np.array(current_embedding, dtype=np.float32)
                    stored_embedding_np = np.array(stored_embedding, dtype=np.float32)

                    similarity = np.dot(current_embedding_np, stored_embedding_np) / \
                                    (np.linalg.norm(current_embedding_np) * np.linalg.norm(stored_embedding_np))
                    
                    print(f"Comparing with {name}, Similarity: {similarity:.4f}")
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
            
            confidence = best_similarity if best_match != "Unknown" else 0.0
            return {"name": best_match, "confidence": confidence}

        except ValueError:
            # This is expected when a face is not found in the image.
            return {"name": "Unknown", "confidence": 0.0}
        except Exception as e:
            print(f"An unexpected error occurred in recognize_face: {e}")
            return {"name": "Unknown", "confidence": 0.0}
