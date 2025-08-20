import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
import os

# Define output path
output_pdf = "aruco_markers_5x5.pdf"

# Marker dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
#parameters = cv2.aruco.DetectorParameters()
#detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# IDs for the four corners
marker_ids = [0, 1, 2, 3]

# Generate marker images
marker_size = 800  # pixels
marker_images = []
for marker_id in marker_ids:
    #marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)

    marker_img = aruco_dict.generateImageMarker(marker_id, marker_size)
    #cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size, marker_img, 1)
    marker_images.append(marker_img)

# Save markers as temporary PNGs
temp_files = []
for i, img in enumerate(marker_images):
    filename = f"./aruco_imgs/marker_{i}.png"
    cv2.imwrite(filename, img)
    temp_files.append(filename)

# Create PDF
c = canvas.Canvas(output_pdf, pagesize=A4)
width, height = A4

# Place markers on each corner of the page
margin = 0.5 * cm
marker_size_cm = 1 * cm  # marker size on paper

positions = [
    (margin, height - margin - marker_size_cm),  # top-left
    (width - margin - marker_size_cm, height - margin - marker_size_cm),  # top-right
    (margin, margin),  # bottom-left
    (width - margin - marker_size_cm, margin)  # bottom-right
]

for pos, file in zip(positions, temp_files):
    c.drawImage(file, pos[0], pos[1], marker_size_cm, marker_size_cm)

c.showPage()
c.save()