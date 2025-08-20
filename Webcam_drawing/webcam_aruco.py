import cv2
import numpy as np

import matplotlib.pyplot as plt

# Load predefined dictionary of ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def detect_aruco(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)
    
    # If markers are detected, draw them
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    return frame

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Detect ArUco markers
        frame = detect_aruco(frame)
        
        # Display the resulting frame
        cv2.imshow('Webcam', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
