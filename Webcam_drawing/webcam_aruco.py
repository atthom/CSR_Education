import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import DiffusionPipeline
from controlnet_aux import HEDdetector, MLSDdetector

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

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

def calculate_perspective_transform(corners):
    # Sort corners based on their position (top-left, top-right, bottom-right, bottom-left)
    sorted_corners = sorted(corners, key=lambda x: (x[0][0][1], x[0][0][0]))  # Sort by y, then x
    tl, tr, br, bl = sorted_corners

    # Define the dimensions of the output image (19cm x 27.7cm)
    width = 1970  # 19.7cm * 100 pixels/cm
    height = 2770  # 27.7cm * 100 pixels/cm

    # Define the destination points
    dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

    # Get the source points from the detected corners
    src_pts = np.array([tl[0][0], tr[0][0], br[0][0], bl[0][0]], dtype=np.float32)

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return M, (width, height)

def apply_perspective_transform(image, M, output_size):
    # Apply the perspective transform
    result = cv2.warpPerspective(image, M, output_size)
    return result

def preprocess_sketch(extracted, corners):
    # Crop the image to remove markers
    cropped = crop_image(extracted, corners)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 4)
    
    # Invert the colors
    #inverted = cv2.bitwise_not(thresh)
    
    # Normalize to 0-1 range
    normalized = thresh.astype(np.float32) / 255.0
    
    # Convert to PIL Image
    pil_image = Image.fromarray((normalized * 255).astype(np.uint8))
    
    return gray

def load_controlnet_model():
    checkpoint = "lllyasviel/control_v11p_sd15_scribble"
    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    #pipe.enable_model_cpu_offload()
    return processor, pipe

def generate_image_from_sketch(sketch, processor, pipe):
    control_image = processor(sketch, scribble=True)
    
    prompt = "old tree with falling leafs on a little hill and a cute flower on the right."
    generator = torch.manual_seed(0)
    image = pipe(prompt, num_inference_steps=50, generator=generator, image=control_image).images[0]

    return image

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load the ControlNet model
    try:
        preprocessor, pipe = load_controlnet_model()
        print(f"ControlNet model loaded successfully. Using device: {pipe.device}")
    except Exception as e:
        print(f"Error loading the ControlNet model: {str(e)}")
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
            # Create mask and extract pixels
            mask = create_polygon_mask(frame, corners)
            extracted = extract_pixels(frame, mask)
            
            # Calculate perspective transform
            #M, output_size = calculate_perspective_transform(corners)
            
            # Apply perspective transform to the extracted image
            #transformed_extracted = apply_perspective_transform(extracted, M, output_size)
            #cv2.imshow('Transformed Extracted', transformed_extracted)
            
            # Preprocess the transformed extracted sketch
            sketch = preprocess_sketch(extracted, corners)
            
            # Convert PIL Image back to OpenCV format for display
            sketch_cv = cv2.cvtColor(np.array(sketch), cv2.COLOR_RGB2BGR)
            cv2.imshow('Preprocessed Sketch', sketch)
            cv2.imshow('Preprocessed sketch_cv', sketch_cv)
            
            # Generate image from sketch
            generated_image = generate_image_from_sketch(sketch, preprocessor, pipe)
            
            # Convert PIL Image to OpenCV format
            generated_cv = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
            cv2.imshow('Generated', generated_cv)
            
            # Visualize the mask
            cv2.imshow('Mask', mask)
        else:
            pass
            #cv2.destroyWindow('Transformed Extracted')
            #cv2.destroyWindow('Preprocessed Sketch')
            #cv2.destroyWindow('Generated')
            #cv2.destroyWindow('Mask')
        
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
