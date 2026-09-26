# ♻️ Smart Waste Detection and Classification using YOLOv8

A deep learning project that detects and classifies waste items (Cardboard, Metal, Paper, Plastic) in images using YOLOv8 object detection. Built as a B.Tech CSE (Data Science) third-year project.

## Overview

This project trains a YOLOv8 object detection model to identify and classify waste materials, supporting automated waste sorting and recycling efforts. It includes a Streamlit-based demo app for real-time image-based detection.

## Dataset

[GARBAGE CLASSIFICATION 4](https://universe.roboflow.com/recycling-vs-waste/garbage-classification-4-oklrj) from Roboflow Universe — 4,178 images across 4 classes: Cardboard, Metal, Paper, Plastic. Exported in YOLOv8 format.

The dataset itself is not included in this repository due to size. To reproduce training, download the dataset from the link above and place it according to the folder structure below.

## Folder Structure

\\\
smart-waste-detection/
├── data/                   # dataset (images + labels), gitignored
├── data.yaml               # YOLO class/path config
├── notebooks/               # training experiments
├── models/
│   └── best.pt             # trained model weights
├── scripts/
│   ├── train.py            # training script
│   ├── detect.py           # inference script
│   └── split_dataset.py    # class-balanced train/val/test splitter
├── results/
│   ├── plots/               # training curves, confusion matrix
│   └── sample_predictions/  # example detection outputs
├── app/
│   └── app.py               # Streamlit demo app
├── requirements.txt
└── README.md
\\\

## Setup

\\\ash
pip install -r requirements.txt
\\\

## Training

Training was done on Google Colab (free GPU). To retrain:

\\\ash
python scripts/train.py
\\\

## Running Inference

\\\ash
python scripts/detect.py --source data/images/test
\\\

## Demo App

\\\ash
streamlit run app/app.py
\\\

Upload an image and the app will detect and classify waste items with bounding boxes.

## Results

Trained YOLOv8n for 50 epochs on a class-balanced train/val/test split (70/20/10).

| Class      | mAP50  |
|------------|--------|
| Cardboard  | 79.2%  |
| Metal      | 88.0%  |
| Paper      | 85.9%  |
| Plastic    | 90.8%  |
| **Overall**| **86.0%** |

See \
esults/plots/\ for training curves and confusion matrix.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
