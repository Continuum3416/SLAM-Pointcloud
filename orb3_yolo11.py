import cv2
import subprocess
from ultralytics import YOLO
import time
import os


# Paths
keyframe_file = "./trajectory/MyKeyFrameTrajectoryTUMFormat.txt"  # Replace with your keyframe file path
detection_file_path = "./yolo_data/yolo_detection.txt"          # File for YOLO detections
video_path = "/home/autonomy/Dev/ORB_SLAM3/Examples/Monocular/lab_video1.mp4"

# Start ORB-SLAM3 in a subprocess
orbslam_process = subprocess.Popen(
    ["/home/autonomy/Dev/ORB_SLAM3/Examples/Monocular/myvideo"], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE
)

# Initialize YOLOv11
model = YOLO("yolo11n.pt") 
cap = cv2.VideoCapture(video_path)

# Open a file to save YOLO detections
with open(detection_file_path, "w") as detection_file:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv11 on the frame
        results = model(frame)

        # Get frame timestamp
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert milliseconds to seconds

        # Save YOLO detections
        for result in results[0].boxes:
            cls = int(result.cls.cpu().numpy()[0])  # Get the class label
            conf = result.conf.cpu().numpy()[0]  # Get confidence score
            class_name = model.names[cls]  # Get the class name from the model
            detection_file.write(f"{timestamp}, {class_name}, {conf:.2f}\n")
        detection_file.flush()  # Ensure immediate write to file

        # Annotate and display the frame
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 + ORB-SLAM3", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Wait for ORB-SLAM3 process to complete
orbslam_process.communicate()
orbslam_process.terminate()