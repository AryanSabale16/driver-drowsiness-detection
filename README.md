# Driver Drowsiness Detection System

A real-time AI-powered driver monitoring system for detecting signs of driver fatigue and drowsiness using computer vision and deep learning.

## Features

The system will monitor:

- Eye closure
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Yawning
- Blink frequency
- PERCLOS
- Head pose
- Driver alertness
- Drowsiness events

## System Architecture

Camera / Video Feed  
↓  
YOLO + MediaPipe  
↓  
Facial & Behavioural Metrics  
↓  
Temporal Analysis  
↓  
Drowsiness Scoring Engine  
↓  
Alert System  
↓  
FastAPI Backend  
↓  
React Dashboard

## Technology Stack

### AI / Computer Vision

- Python
- PyTorch
- Ultralytics YOLO
- MediaPipe
- OpenCV
- NumPy

### Backend

- FastAPI
- WebSockets

### Frontend

- React
- Vite
- Tailwind CSS
- Recharts

### Database

- SQLite

## Project Status

🚧 Currently under development.

### Progress

- [x] Development environment
- [x] CUDA / GPU configuration
- [x] Initial project architecture
- [ ] MediaPipe facial landmarks
- [ ] Eye state detection
- [ ] Yawn detection
- [ ] Head pose estimation
- [ ] YOLO integration
- [ ] Drowsiness scoring
- [ ] FastAPI backend
- [ ] React dashboard
- [ ] Alert system
- [ ] Evaluation
- [ ] Deployment

## Hardware

Development and testing currently performed using:

- NVIDIA GeForce GTX 1650
- CUDA-enabled PyTorch

## Disclaimer

This project is intended for research and educational purposes and should not be considered a replacement for certified automotive driver-monitoring safety systems.