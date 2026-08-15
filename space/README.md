---
title: ADAS Perception Demo
emoji: 🚦
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

Calibrated camera-LIDAR fusion + a KITTI-fine-tuned YOLO11 detector.
Upload any image for detection, or pick a real KITTI frame to see the full
calibrated fusion pipeline — LIDAR-derived distance per detected object.

Model: [mokshhere/adas-kitti-yolo11m](https://huggingface.co/mokshhere/adas-kitti-yolo11m)
Code: [ADAS-Object-Detection-and-Collision-Avoidance](https://github.com/ishaannk/ADAS-Object-Detection-and-Collision-Avoidance)
