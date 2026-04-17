# Aerial Guardian — Drone-Based Person Detection and Tracking System

## Project Overview

**Aerial Guardian** is a computer vision system designed to detect and track persons in aerial drone footage. The system processes image sequences from the VisDrone dataset, detects persons using a deep learning model, assigns unique IDs to each detected individual, and visualizes movement trajectories along with real-time performance metrics such as Frames Per Second (FPS).

This project demonstrates a complete end-to-end pipeline including detection, tracking, visualization, and video generation suitable for academic submission and technical evaluation.

---

## Key Features

* Person detection using YOLOv8 deep learning model
* 3×3 tiling strategy to improve detection of small objects
* Custom IoU-based multi-object tracking system
* Unique ID assignment for each tracked person
* Movement trail visualization
* Duplicate detection removal using IoU thresholding
* Real-time FPS display on video output
* Automatic video generation for each sequence
* Works on CPU-based systems

---

## System Architecture

Input Images → Tiling Detection → Duplicate Removal → Tracking → Visualization → Video Output

---

## Technologies Used

* Python 3.x
* OpenCV
* NumPy
* Ultralytics YOLOv8
* VisDrone Dataset

---

## Hardware Requirements

Minimum:

* Processor: Intel Core i5 (11th Gen recommended)
* RAM: 8 GB
* Storage: 10 GB free space
* Operating System: Windows 10 / 11

Tested System:

* CPU: Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
* RAM: 8 GB
* GPU: Not required (CPU execution)

---

## Software Requirements

Install required Python libraries:

pip install ultralytics opencv-python numpy

Download YOLO model:

from ultralytics import YOLO
YOLO("yolov8n.pt")

---

## Dataset Structure

VisDrone2019-MOT-val/

```
sequences/

    seq1/
        0000001.jpg
        0000002.jpg
        ...
```

Outputs will be generated in:

outputs/

```
videos/
    seq1.mp4
    seq2.mp4
```

---

## How to Run the Project

Step 1 — Place dataset in project folder

Step 2 — Ensure model file exists:

```
yolov8n.pt
```

Step 3 — Run the script:

python main.py

Step 4 — Check output videos:

outputs/videos/

---

## Output Description

Each generated video contains:

* Green bounding boxes around detected persons
* Unique tracking IDs
* Red movement trajectory lines
* FPS displayed in the top-left corner

Example console output:

Processing seq1...
Average FPS: 1

DONE — Final Improved Output Generated

---

## Algorithm Details

### 1. Detection

The YOLOv8 model detects persons in each frame. A 3×3 tiling strategy is applied to improve detection accuracy for small objects in aerial imagery.

### 2. Duplicate Removal

Bounding boxes with high overlap are filtered using Intersection over Union (IoU) thresholding.

### 3. Tracking

Each detected object is matched to an existing track using IoU similarity. If no match is found, a new track ID is assigned.

### 4. Visualization

The system draws:

* Bounding boxes
* Object IDs
* Movement trails
* Real-time FPS

---

## Performance Metrics

The system measures:

* Frame Processing Speed (FPS)
* Detection Stability
* Tracking Consistency

Typical CPU Performance:

10 to 15 FPS on Intel i5 processor

---

## Engineering Trade-offs

| Component          | Choice      | Reason                        |
| ------------------ | ----------- | ----------------------------- |
| Model              | YOLOv8 Nano | Fast CPU inference            |
| Tracking           | IoU-based   | Lightweight implementation    |
| Detection Strategy | 3×3 Tiling  | Better small object detection |
| Hardware           | CPU         | No GPU dependency             |

---

## Future Improvements

* GPU acceleration support
* Advanced tracking algorithm (DeepSORT or ByteTrack)
* Real-time webcam integration
* Evaluation metrics (MOTA, IDF1)
* Multi-class object detection

---

## Author

Name: Ashish Yadav

Institution: National Institute of Technology Jamshedpur

Course / Department: Mtech-Communication System Engineering/ ECE

---

## License

This project is developed for academic and educational purposes.
