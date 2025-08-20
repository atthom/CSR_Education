import pandas as pd
import numpy as np

import time
import cv2
from ultralytics import YOLO


def webcam_loop():
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    t = time.time()

    while rval:
        #cv2.imshow("preview", frame)

        rval, frame = vc.read()
        key = cv2.waitKey(20)

        if time.time() - t > 0.1:
            t = time.time()
            seg = model(frame)
            seg[0].show()
            #print(seg)
            #cv2.imshow("preview", seg)

        if key == 27: # exit on ESC
            break
        if key == 13:
            print("Enter")

    vc.release()
    cv2.destroyWindow("preview")

def only_yolo():
    results = model(source="0", stream=True)

def yolo_stream():
    cap = cv2.VideoCapture(0)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO inference on the frame
            results = model.track(frame, tracker="botsort.yaml")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("AI Segmentation", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
    
    cap.release()
    cv2.destroyAllWindows()


model = YOLO("yolo11s-pose.pt")
#source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#model.predict(source=source, save=True)
#only_yolo()
yolo_stream()