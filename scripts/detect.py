"""
Runs inference using the trained YOLOv8 waste detection model on a folder
of images (e.g. your test set, or any new images you want to test on).

Usage (run from project root, A:\\YOLOWaste):
    python scripts/detect.py
    python scripts/detect.py --source data/images/test
    python scripts/detect.py --source path/to/some/folder --conf 0.5
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 waste detection inference")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images/test",
        help="Folder of images (or single image path) to run detection on"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (0-1)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/sample_predictions",
        help="Where to save annotated output images"
    )
    args = parser.parse_args()

    print(f"Loading model from {args.weights}...")
    model = YOLO(args.weights)

    print(f"Running detection on: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project=args.save_dir,
        name="predictions",
        exist_ok=True
    )

    print(f"\nDone. Processed {len(results)} image(s).")
    print(f"Annotated images saved to: {args.save_dir}/predictions/")

    # Quick summary of what was detected across all images
    class_names = model.names
    detection_counts = {}
    for r in results:
        for cls_id in r.boxes.cls.tolist():
            cls_name = class_names[int(cls_id)]
            detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1

    print("\nDetection summary:")
    if detection_counts:
        for cls_name, count in sorted(detection_counts.items()):
            print(f"  {cls_name}: {count} detected")
    else:
        print("  No objects detected.")


if __name__ == "__main__":
    main()