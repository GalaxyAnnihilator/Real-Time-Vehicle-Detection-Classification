# YOLOv8 + CNN Real-Time Vehicle Detection & Classification

## Overview
This project combines **YOLOv8** and **CNN** to detect and classify vehicles from images or video streams. It follows a **cascading** ensemble model approach:
1. **YOLOv8** detects vehicles and crops them.
2. **CNN** classifies the detected vehicle type (e.g., car, truck, motorcycle, bus, bicyle).

## 🎯 Features
✅ Real-time vehicle detection and classification  
✅ Modular design (YOLO for detection, CNN for classification)  
✅ Supports multiple vehicle types  
✅ Easy to train on custom datasets  
✅ Deployable on cloud or edge devices  

## 📂 Project Structure
```
├── dataset/
│   ├── train/
│   │   ├── car/
│   │   ├── truck/
│   │   ├── motorcycle/
│   │   ├── bus/
│   │   ├── bicyle/
│   ├── valid/
│   │   ├── car/
│   │   ├── truck/
│   │   ├── motorcycle/
│   │   ├── bus/
│   │   ├── bicyle/
├── models/
│   ├── best.pt  # Trained YOLOv8 model
│   ├── vehicle_cnn.h5  # Trained CNN model
├── scripts/
│   ├── train_yolo.ipynb
│   ├── train_cnn.ipynb
│   ├── detect_and_classify.py
├── requirements.txt
├── README.md
```

## 📦 Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/GalaxyAnnihilator/Real-Time-Vehicle-Detection-Classification
cd Real-Time-Vehicle-Detection-Classification
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Download YOLOv8 weights
```bash
pip install ultralytics
```

## 🚀 Training
### **Train YOLOv8 for Vehicle Detection**
train_yolo.ipynb

### **Train CNN for Vehicle Classification**
train_cnn.ipynb

## Inference: Detect & Classify Vehicles
Run inference on an image:
```bash
python scripts/detect_and_classify.py --image test_image.jpg
```

## Run Real-Time Detection
Run the live camera detection:
```bash
python scripts/real_time_detection.py
```

## 📊 Results
The output will be an image with **bounding boxes and class labels**.

## 🛠 Future Improvements
🔹 Improve CNN model with transfer learning (ResNet, EfficientNet)  
🔹 Optimize for real-time deployment (TensorRT, OpenVINO)  
🔹 Expand dataset for better generalization  

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Feel free to fork and contribute! Open an issue if you find bugs or have suggestions.

---
🚀 **Developed by Tran Minh Duong**

