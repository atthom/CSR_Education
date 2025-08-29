import cv2
import numpy as np
import os

# Import necessary functions from webcam_aruco.py
from webcam_aruco import detect_aruco, create_polygon_mask, extract_pixels, crop_image

# Create output directory if it doesn't exist
output_dir = "canny_results"
os.makedirs(output_dir, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Capture frame with ArUco markers
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        cap.release()
        exit()

    frame, corners, ids = detect_aruco(frame)
    
    if ids is not None and len(ids) >= 4:
        mask = create_polygon_mask(frame, corners)
        extracted = extract_pixels(frame, mask)
        cropped = crop_image(extracted, corners)
        break

    cv2.imshow('Capture Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Capture cancelled by user")
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Convert cropped image to grayscale
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

total_iterations = 51 * 51  # 0 to 250 with step 5 for both thresholds
current_iteration = 0

for lower in range(30, 200, 5):
    for upper in range(40, 160, 5):
        # Apply Canny edge detection
        edges = cv2.Canny(gray, lower, upper)

        # Invert colors (white to black and black to white)
        inverted_edges = cv2.bitwise_not(edges)

        # Save the resulting image
        filename = f"{output_dir}/canny_lower{lower}_upper{upper}.png"
        cv2.imwrite(filename, inverted_edges)

        # Update and print progress
        current_iteration += 1
        progress = (current_iteration / total_iterations) * 100
        print(f"Progress: {progress:.2f}%", end="\r")

print("\nProcessing complete. Results saved in the 'canny_results' directory.")
