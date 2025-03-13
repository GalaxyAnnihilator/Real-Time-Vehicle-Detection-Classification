import cv2
from ultralytics import YOLO
import numpy as np

# 1. Load pretrained YOLOv8 model (automatically downloads if not found)
model = YOLO('yolov8n.pt')  # 'n'=nano (fastest), 's'=small, 'm'=medium, 'l'=large, 'x'=xlarge

# 2. Define vehicle classes (YOLO COCO dataset classes)
VEHICLE_CLASSES = [2, 3, 5, 7]  # 2=car, 3=motorcycle, 5=bus, 7=truck

# 3. Open video file
video_path = 'rushhour.mp4'
cap = cv2.VideoCapture(video_path)

# 4. Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 5. Create VideoWriter to save results (optional)
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 6. Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error")
        break
    
    # Run YOLO inference (only detect vehicles)
    results = model.predict(
        frame,
        classes=VEHICLE_CLASSES,
        conf=0.3  # Minimum confidence threshold
    )
    
    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()  # Automatic visualization
    
    # Display frame
    cv2.imshow('Vehicle Detection', annotated_frame)
    
    # Save to output video (optional)
    out.write(annotated_frame)
    print("success")
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Cleanup
cap.release()
out.release()  # If saving output
cv2.destroyAllWindows()