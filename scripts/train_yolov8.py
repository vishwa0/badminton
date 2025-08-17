import argparse
from ultralytics import YOLO

def train_model(data_yaml_path, epochs, model_path='yolov8n.pt'):
    """
    Train a YOLOv8 model.

    Args:
        data_yaml_path (str): Path to the data.yaml file.
        epochs (int): Number of training epochs.
        model_path (str): Path to the initial model weights.
    """
    # Load the model
    model = YOLO(model_path)

    # Train the model
    model.train(data=data_yaml_path, epochs=epochs)

    # Save the trained model
    model.save('yolov8n_trained.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data.yaml file.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the initial model weights.')
    args = parser.parse_args()

    train_model(args.data, args.epochs, args.model)