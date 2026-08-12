# Driver Drowsiness Detection System

A real-time AI-powered driver monitoring system designed to detect signs of driver fatigue and drowsiness using computer vision, facial behavioural analysis, temporal analysis, and deep learning.

The system analyzes multiple behavioural indicators including eye closure, blinking, yawning, and head movement to determine the driver's alertness level and provide timely warnings and notifications.

---

## Features

The system monitors multiple indicators of driver behaviour:

- Eye Aspect Ratio (EAR)
- Eye state detection
- Blink detection
- Blink frequency / blink rate
- Eye closure duration
- PERCLOS
- Mouth Aspect Ratio (MAR)
- Mouth state detection
- Yawn detection
- Yawn duration
- Head pose estimation
- Head direction
- Head-down duration
- Temporal behavioural analysis
- Drowsiness event detection
- Driver alertness assessment

### Planned Alert & Notification Features

- Real-time visual warnings
- Audio alarm
- SMS notification
- Registered emergency contact system
- Fleet manager notification
- Family member notification for private vehicles
- GPS location sharing

---

# Technology Stack

## AI / Computer Vision

- Python
- OpenCV
- MediaPipe
- NumPy
- PyTorch
- Ultralytics YOLO
- SciPy

## Behavioural Analysis

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- PERCLOS
- Blink Rate
- Temporal Eye Closure Analysis
- Temporal Yawn Analysis
- Head Pose Estimation

## Backend

- FastAPI
- WebSockets

## Frontend

- React
- Vite
- Tailwind CSS
- Recharts

## Database

- SQLite

## Planned Communication

- SMS notification service
- Registered emergency contacts
- Fleet monitoring notifications

---

# Project Status

🚧 **Currently under active development**

The project is being developed in multiple stages, starting with individual behavioural detection modules and progressing toward an integrated real-time drowsiness intelligence and monitoring platform.

## Development Progress

### Phase 1 — Behaviour Analysis

- [x] Development environment
- [x] CUDA / GPU configuration
- [x] Initial project architecture
- [x] MediaPipe facial landmarks
- [x] Eye landmark detection
- [x] Eye Aspect Ratio (EAR)
- [x] Eye state detection
- [x] Blink detection
- [x] Blink rate calculation
- [x] Eye closure duration
- [x] PERCLOS calculation
- [x] Mouth landmark detection
- [x] Mouth Aspect Ratio (MAR)
- [x] Mouth state detection
- [x] Yawn detection
- [x] Yawn duration analysis
- [x] Head pose estimation
- [x] Head direction detection
- [x] Head-down duration analysis
- [x] Temporal behavioural analysis
- [x] Full behavioural analysis integration

### Phase 2 — Drowsiness Intelligence

- [ ] Multi-signal drowsiness scoring
- [ ] Eye closure + PERCLOS integration
- [ ] Yawning integration
- [ ] Head movement integration
- [ ] Behavioural severity levels
- [ ] Drowsiness event classification
- [ ] False-positive reduction
- [ ] Adaptive / configurable thresholds

### Phase 3 — Model Training

- [ ] Dataset preparation
- [ ] Dataset preprocessing
- [ ] YOLO dataset configuration
- [ ] Model training
- [ ] Model validation
- [ ] Model evaluation
- [ ] Performance benchmarking
- [ ] Trained model integration

### Phase 4 — Alert & Notification System

- [ ] Visual warning system
- [ ] Audio alarm
- [ ] Alarm severity levels
- [ ] Alarm cooldown / event management
- [ ] SMS notification
- [ ] Registered emergency contact
- [ ] Fleet manager notification
- [ ] Family member notification
- [ ] Notification event logging

### Phase 5 — Location & Backend

- [ ] GPS integration
- [ ] Location tracking
- [ ] FastAPI backend
- [ ] WebSocket communication
- [ ] Database integration
- [ ] Driver / vehicle registration
- [ ] Emergency contact management
- [ ] Drowsiness event storage

### Phase 6 — React Dashboard

- [ ] Dashboard UI
- [ ] Live camera feed
- [ ] Driver status
- [ ] Drowsiness status
- [ ] EAR / PERCLOS metrics
- [ ] Blink statistics
- [ ] Yawn statistics
- [ ] Head pose information
- [ ] Alert status
- [ ] Notification history
- [ ] GPS / vehicle location
- [ ] Historical analytics
- [ ] Charts and reports

### Phase 7 — Final Integration & Deployment

- [ ] Complete system integration
- [ ] Backend + AI integration
- [ ] React + backend integration
- [ ] Alert + notification integration
- [ ] GPS integration
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Error handling
- [ ] Deployment configuration
- [ ] Final documentation

---

