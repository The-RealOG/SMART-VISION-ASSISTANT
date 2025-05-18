# SMART-VISION-ASSISTANT
# 🔍 Smart Vision Assistant - Real-Time Object Detection with YOLO

This project is a **real-time object detection system** using a webcam and **YOLOv3-Tiny**, a lightweight variant of the YOLO (You Only Look Once) deep learning model. It uses **OpenCV's DNN module** for inference and supports live detection and annotation of common objects from the COCO dataset.

---

## 📸 Features

- Live webcam feed processing
- Real-time object detection using YOLOv3-Tiny
- Bounding box annotations with confidence scores
- Lightweight and fast — suitable for laptops or Raspberry Pi
- Built with **Python**, **OpenCV**, and **YOLOv3-Tiny**

---

## 📁 Folder Structure
<pre> ```text vision/ ├── yolo/ │ ├── yolov3-tiny.onnx # ONNX model file │ ├── coco.names # COCO dataset class names │ └── yolo_webcam.py # Main Python script ├── README.md ├── .gitignore └── requirements.txt ``` </pre>
