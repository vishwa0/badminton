from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model on the custom dataset
if __name__ == '__main__':
    model.train(
        data='badminton-court-detection-3/data.yaml',
        epochs=50,
        imgsz=640,
        device='0'  # Use CUDA device 0
    )