from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 nano model (small, fast, good for a college project)
    model = YOLO('yolov8n.pt')

    # Train the model on your dataset
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='waste_detector'   # results saved under runs/detect/waste_detector
    )

if __name__ == '__main__':
    main()