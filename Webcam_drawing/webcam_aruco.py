import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


# Load predefined dictionary of ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
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
    
    return frame, corners, ids

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

def preprocess_sketch(extracted):
    # Convert to grayscale
    gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    # Invert colors
    inverted = cv2.bitwise_not(gray)
    # Normalize to 0-1 range
    normalized = inverted.astype(np.float32) / 255.0
    # Convert to PIL Image
    pil_image = Image.fromarray((normalized * 255).astype(np.uint8))
    return pil_image

def generate_image_from_sketch(sketch, pipe):
    # Generate image from sketch
    with torch.no_grad():
        image = pipe(prompt="A detailed colorful image", image=sketch).images[0]
    return image

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load the pre-trained model
    try:
        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model loaded successfully. Using device: {pipe.device}")
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return
    
    frame_count = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to grab frame. Frame count: {frame_count}")
            break
        
        frame_count += 1
        
        # Detect ArUco markers
        frame, corners, ids = detect_aruco(frame)
        
        # If at least 4 markers are detected, create mask, extract pixels and generate image
        if ids is not None and len(ids) >= 4:
            mask = create_polygon_mask(frame, corners)
            extracted = extract_pixels(frame, mask)
            cv2.imshow('Extracted', extracted)
            
            # Preprocess the extracted sketch
            sketch = preprocess_sketch(extracted)
            
            # Generate image from sketch
            generated_image = generate_image_from_sketch(sketch, pipe)
            
            # Convert PIL Image to OpenCV format
            generated_cv = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
            cv2.imshow('Generated', generated_cv)
            
            # Visualize the mask
            cv2.imshow('Mask', mask)
        else:
            cv2.destroyWindow('Extracted')
            cv2.destroyWindow('Generated')
            cv2.destroyWindow('Mask')
        
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
