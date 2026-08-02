import cv2
import mediapipe as mp
import torch
from ultralytics import YOLO

print("=" * 50)
print("Driver Drowsiness Detection Environment Test")
print("=" * 50)

# OpenCV
print(f"OpenCV Version      : {cv2.__version__}")

# MediaPipe
print("MediaPipe           : Installed")

# PyTorch
print(f"PyTorch Version     : {torch.__version__}")

# CUDA
print(f"CUDA Available      : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name            : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version        : {torch.version.cuda}")
else:
    print("Running on CPU")

# YOLO
model = YOLO("yolo11n.pt")
print("YOLO                : Loaded Successfully")

print("=" * 50)
print("Environment Ready!")