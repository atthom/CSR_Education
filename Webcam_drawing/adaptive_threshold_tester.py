import cv2
import numpy as np
from PIL import Image
import os


# Load predefined dictionary of ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def detect_aruco(frame):
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)
    
    return corners, ids

def create_polygon_mask(frame, corners):
    # Create a mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    # Extract the corners of all markers and flatten them into a single array
    polygon_points = np.concatenate([corner.reshape(-1, 2) for corner in corners])
    
    # Find the convex hull of all points to create a polygon
    hull = cv2.convexHull(polygon_points.astype(np.int32))
    
    # Fill the polygon
    cv2.fillConvexPoly(mask, hull, 255)
    
    return mask

def extract_pixels(frame, mask):
    # Extract pixels
    extracted = cv2.bitwise_and(frame, frame, mask=mask)
    
    return extracted

def crop_image(image, corners):
    # Find the bounding rectangle of the polygon
    points = np.concatenate(corners).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(points)
    
    # Add a small margin (e.g., 5% of width/height) to ensure we don't crop too tightly
    margin_x = -int(w * 0.1)
    margin_y = -int(h * 0.1)
    
    # Crop the image, ensuring we don't go out of bounds
    cropped = image[max(0, y-margin_y):min(image.shape[0], y+h+margin_y),
                    max(0, x-margin_x):min(image.shape[1], x+w+margin_x)]
    
    return cropped

def preprocess_sketch(extracted, corners, block_size, C):
    # Crop the image to remove markers
    cropped = crop_image(extracted, corners)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive Gaussian thresholding
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    
    return bw

def generate_threshold_images(frame, corners, output_dir):
    # Create mask and extract pixels
    mask = create_polygon_mask(frame, corners)
    extracted = extract_pixels(frame, mask)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different parameters
    for block_size in range(3, 52, 2):  # Odd values from 3 to 51
        for C in range(4, 11):  # Values from -10 to 10
            # Preprocess the sketch with current parameters
            bw = preprocess_sketch(extracted, corners, block_size, C)
            
            # Save the image
            filename = f"threshold_block{block_size}_C{C}.png"
            cv2.imwrite(os.path.join(output_dir, filename), bw)

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Detect ArUco markers
        corners, ids = detect_aruco(frame)
        
        # If at least 4 markers are detected, generate threshold images
        if ids is not None and len(ids) >= 4:
            generate_threshold_images(frame, corners, "threshold_outputs")
            print("Threshold images generated. Check the 'threshold_outputs' directory.")
            break
        
        # Display the frame
        cv2.imshow('Webcam', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
