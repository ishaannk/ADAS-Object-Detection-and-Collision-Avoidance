# ADAS-Object-Detection-and-Collision-Avoidance


This project focuses on **Advanced Driver Assistance Systems (ADAS)** using **YOLOv5** for object detection and collision avoidance. It processes images from the **KITTI dataset**, applies real-time object detection, and integrates sensor fusion for improved safety.

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

## Running the Project

### Instructions to Clone this Repository
1. Open a terminal or command prompt.
2. Run the following command to clone the repository:
   ```bash
   git clone https://github.com/your-repo/ADAS-Object-Detection.git
   ```
3. Navigate into the project directory:
   ```bash
   cd ADAS-Object-Detection
   ```

### Steps to Run the Project

1. Clone the repository and navigate to the folder:
   ```bash
   git clone https://github.com/your-repo/ADAS-Object-Detection.git
   cd ADAS-Object-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to process images and visualize detection:
   ```bash
   jupyter notebook ADAS_Exercise.ipynb
   ```

## Future Scope

- Improve accuracy with fine-tuned YOLO models.
- Integrate real-time streaming for live ADAS applications.
- Deploy on edge devices for low-latency performance.

