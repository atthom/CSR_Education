from alphai import *
import numpy as np
import cv2
import os

# Extraction de l image du robot AlphAI
connect_wifi()
im_cam = get_camera()
im_cam = np.array(im_cam)
disconnect()
print(im_cam)

# Enregistrement de l image (facultatif)
directory = 'C:/Users/sap/Documents/Stage_Lou_Marze/programmation/fonction recompense/'
nb_pix = len(im_cam) * len(im_cam[0])
print(nb_pix)
os.chdir(directory)
im_to_convert =np.float32(im_cam)
im_to_write = cv2.cvtColor(im_to_convert, cv2.COLOR_BGR2RGB)
cv2.imwrite("image_capteur.png",im_to_write)



