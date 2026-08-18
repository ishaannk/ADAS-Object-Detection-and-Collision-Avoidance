# ADAS-Object-Detection-and-Collision-Avoidance


This project focuses on **Advanced Driver Assistance Systems (ADAS)** using **YOLOv5** for object detection and collision avoidance. It processes images from the **KITTI dataset**, applies real-time object detection, and integrates sensor fusion for improved safety.
Improved Version : https://github.com/ishaannk/adas-rebuild
## Use Cases

- **Object Detection**: Identifies vehicles, pedestrians, and obstacles using YOLOv5.
- **Collision Avoidance**: Computes safe distances and potential collision risks.
- **Sensor Fusion**: Merges data from camera and LIDAR for enhanced accuracy.
- **Autonomous Driving Research**: Aids in developing intelligent navigation systems.

## Explanations

### 1. Environment Setup

The following dependencies are installed:

```bash
pip install tensorflow-datasets torch ultralytics opencv-python matplotlib numpy
```

### 2. Dataset and Model

- **Dataset**: KITTI dataset (used for training/testing)
- **Model**: YOLOv5 (pre-trained on COCO for object detection)

### 3. Data Preprocessing & Visualization

- Load and visualize raw images from the dataset.
- Extract relevant information from LIDAR sensor data.

### 4. Object Detection & Collision Avoidance

- Use **YOLOv5** to detect objects in real-time.
- Apply bounding boxes and confidence scores to detect pedestrians, vehicles, and other objects.
- Compute distances and potential collision risks.

### 5. Multiple Example Scenarios

- The project runs detection on multiple test cases, demonstrating object detection and collision avoidance in real-world scenarios.




<img width="1420" height="488" alt="image" src="https://github.com/user-attachments/assets/0b0715f0-d339-4a51-8f3f-d23236a50a96" />

<img width="1449" height="508" alt="image" src="https://github.com/user-attachments/assets/ee6279e9-0bed-4db8-ab00-8ecd23898b17" />
<img width="1603" height="429" alt="image" src="https://github.com/user-attachments/assets/698e7ac5-f2cd-4ec3-b3da-db558befa228" />


## Future Scope

- Improve accuracy with fine-tuned YOLO models.
- Integrate real-time streaming for live ADAS applications.
- Deploy on edge devices for low-latency performance.

