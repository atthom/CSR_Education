import cv2
import os
import numpy as np

directory = 'C:/Users/sap/Documents/Stage_Lou_Marze/programmation/fonction recompense/'
os.chdir(directory)
image = cv2.imread("image_capteur.png")
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

reward = 0

# Split into 2 cases, centered green or external green
dimension = image.shape[:2]
with_div_4 = (dimension[1]//4)
# Creation of the mask
center_mask = np.zeros(dimension, np.uint8) # .shape[:2] get the height and width of the image
ext_mask = np.ones(dimension, np.uint8) * 255 # initialisation of the external mask with black pixels
center_mask[:, with_div_4 :3*with_div_4] = 255
ext_mask[:, with_div_4 :3*with_div_4] = 0

# Display of the centered mask
cv2.imshow("center_mask", center_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display of the external mask
cv2.imshow("mask_ext", ext_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Application of the masks to the picture
centered_mask_image = cv2.bitwise_and(image, image, mask = center_mask)
ext_mask_image = cv2.bitwise_and(image, image, mask = ext_mask)

# Display the masked pictures
cv2.imshow("centered_mask_image", centered_mask_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("ext_mask_image", ext_mask_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Creation of a green mask
lower_green = np.array([0,100,0], dtype = np.uint8) # threshold to set if the green is not detected or too much detected
upper_green = np.array([117,255,117], dtype = np.uint8)
# the mask contains only the green pixels of the picture
green_centered_masked_image = cv2.inRange(centered_mask_image, lower_green, upper_green)
green_ext_masked_image = cv2.inRange(ext_mask_image, lower_green, upper_green)
# Display the green masked image
cv2.imshow("green_masked_image", green_centered_masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Counting the number of green pixels in the 2 cases
nb_centered_pixel = np.count_nonzero(green_centered_masked_image)
reward += 0.0002* nb_centered_pixel # set the values to change the reward
nb_ext_pixel = np.count_nonzero(green_ext_masked_image)
reward += 0.00009*nb_ext_pixel

# Mask the blue pixels around the green one
print(reward)
