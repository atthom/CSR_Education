import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from controlnet_aux import PidiNetDetector, HEDdetector
from diffusers.utils import load_image
from huggingface_hub import HfApi
from pathlib import Path
import os
import threading
import queue
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# Load predefined dictionary of ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

import torch._dynamo
torch._dynamo.config.suppress_errors = True

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
    #ddim_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0",
        torch_dtype=torch.float16
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


    #pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.enable_model_cpu_offload()

    processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
    
    return processor, pipe

def generate_image_from_sketch(sketch, processor, pipe):
    
    #print(type(control_image), type(sketch))
    control_image = sketch
    prompt = "good quality, photorealistic, professional, high res, 4k"
    negative_prompt = 'longbody, too much detail, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
    
    controlnet_conditioning_scale = 0.8  # Slightly reduced from 1.0
    
    # Resize the control image to 1024x1024 or maintain aspect ratio
    width, height = control_image.size
    ratio = np.sqrt(1024. * 1024. / (width * height)) 
    new_width, new_height = int(width * ratio), int(height * ratio)
    control_image = control_image.resize((new_width, new_height))
    control_image = processor(control_image, scribble=True)

    image = result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=control_image,
        num_inference_steps=20,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.8,
        added_cond_kwargs={}  # ✅ Prevent NoneType crash
    ).images[0]

    return image

class ImageGenerator(threading.Thread):
    def __init__(self, preprocessor, pipe):
        threading.Thread.__init__(self)
        self.preprocessor = preprocessor
        self.pipe = pipe
        self.queue = queue.Queue(maxsize=1)  # Limit queue size to 1
        self.result = None
        self.running = True
        self.processing = False
        self.new_result = False

    def run(self):
        while self.running:
            try:
                if not self.processing:
                    sketch = self.queue.get(timeout=1)
                    self.processing = True
                    with torch.no_grad():
                        self.result = generate_image_from_sketch(sketch, self.preprocessor, self.pipe)
                    self.processing = False
                    self.new_result = True
            except queue.Empty:
                continue

    def stop(self):
        self.running = False

    def is_processing(self):
        return self.processing

    def get_result(self):
        if self.new_result:
            self.new_result = False
            return self.result
        return None

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
    
    # Create and start the image generator thread
    #image_generator = ImageGenerator(preprocessor, pipe)
    #image_generator.start()
    
    frame_count = 0
    last_sketch = None
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
            
            # Preprocess the extracted sketch
            sketch = preprocess_sketch(extracted, corners)
            cv2.imshow('Preprocessed Sketch', sketch)
            
            # Queue the sketch for image generation if it's different from the last one and not currently processing
            if False:
                if not np.array_equal(sketch, last_sketch) and not image_generator.is_processing():
                    try:
                        image_generator.queue.put_nowait(sketch)
                        last_sketch = sketch
                    except queue.Full:
                        pass  # Queue is full, skip this frame
                
                # Check if a new generated image is available
                generated_image = image_generator.get_result()
                if generated_image is not None:
                    generated_cv = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
                    cv2.imshow('Generated', generated_cv)
                    input()
                    
            sketch = Image.fromarray(np.uint8(sketch)).convert('RGB')
            generated_image = generate_image_from_sketch(sketch, preprocessor, pipe)
            generated_cv = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)
            cv2.imshow('Generated', generated_cv)
            
            # Visualize the mask
            #cv2.imshow('Mask', mask)
        else:
            pass
            #cv2.destroyWindow('Preprocessed Sketch')
            #cv2.destroyWindow('Preprocessed sketch_cv')
            #cv2.destroyWindow('Generated')
            #cv2.destroyWindow('Mask')
        
        # Display the resulting frame
        cv2.imshow('Webcam', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Stop the image generator thread
    image_generator.stop()
    image_generator.join()
    
    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
