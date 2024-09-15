import cv2
import numpy as np
from iofile import *

# Load the two images
img1 = cv2.imread('result/ascii8.jpg')  
img2 = cv2.imread('result/saturation_edge_ascii_8.jpg')  


gray1 = desat_graysc(img1,True)
gray2 = desat_graysc(img2,True)

# Thresholding edge_image to create a mask of edges
_, mask = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)

# Invert the mask to get the non-edge areas
mask_inv = cv2.bitwise_not(mask)

# Use mask to keep edges from Image 2
edges_from_img2 = cv2.bitwise_and(gray2, gray2, mask=mask)

# Use inverted mask to keep  non-edge parts from Image 1
filled_from_img1 = cv2.bitwise_and(gray1, gray1, mask=mask_inv)

# Combine the edges and filled areas
combined_image = cv2.add(edges_from_img2, filled_from_img1)

display(combined_image)
cv2.imwrite('result/combined_ascii8.jpg', combined_image)